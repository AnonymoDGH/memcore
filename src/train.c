#include "train.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "infer.h"
#include "tasks.h"
#include "tensor.h"

#define PCAP 48
#define BUF_CAP 2048
#define MASTERY 0.9f
#define EVAL_N 64
#define STAR_N 32

typedef struct {
    char p[BUF_CAP][PCAP];
    int n, head;
} Ring;

static Ring g_hard; /* problems the model failed: hard-example mining */
static Ring g_self; /* frontier problems it solved itself (STaR) */

static void ring_push(Ring *r, const char *p) {
    snprintf(r->p[r->head], PCAP, "%s", p);
    r->head = (r->head + 1) % BUF_CAP;
    if (r->n < BUF_CAP) r->n++;
}

static int *levels(Model *M) {
    for (int t = 0; t < T_COUNT; t++)
        if (M->meta[t] < 1) M->meta[t] = 1;
    return M->meta;
}

static void sample_example(Model *M, char *prompt, char *ans) {
    float r = rng_uniform();
    if (r < 0.15f && g_hard.n) {
        snprintf(prompt, PCAP, "%s", g_hard.p[rng_int(g_hard.n)]);
    } else if (r < 0.25f && g_self.n) {
        snprintf(prompt, PCAP, "%s", g_self.p[rng_int(g_self.n)]);
    } else {
        int t = rng_int(T_COUNT), L = levels(M)[t];
        int lv = rng_uniform() < 0.6f ? L : 1 + rng_int(L);
        task_gen(t, lv, prompt, PCAP);
    }
    task_solve(prompt, ans, PCAP);
}

static void fill_row(Model *M, const TrainOpts *o, int *tok, int *tgt) {
    int T = o->T, s[MC_MAXT + 1], len = 0;
    char m[MC_MAXT + 1];
    if (o->text && o->text_len > (size_t)T + 1 && rng_uniform() < o->text_mix) {
        size_t st = (size_t)(rng_u64() % (o->text_len - (size_t)T - 1));
        for (int i = 0; i <= T; i++) { s[i] = o->text[st + i]; m[i] = 1; }
    } else {
        s[len] = '\n'; m[len++] = 0;
        while (len <= T) {
            char p[PCAP], a[PCAP];
            sample_example(M, p, a);
            for (char *c = p; *c && len <= T; c++) { s[len] = (unsigned char)*c; m[len++] = 0; }
            for (char *c = a; *c && len <= T; c++) { s[len] = (unsigned char)*c; m[len++] = 1; }
            if (len <= T) { s[len] = '\n'; m[len++] = 1; }
        }
    }
    for (int i = 0; i < T; i++) {
        tok[i] = s[i];
        tgt[i] = m[i + 1] ? s[i + 1] : -1;
    }
}

static float eval_level(Model *M, KV *kv, int task, int level, int n,
                        int adaptive, int votes, Ring *fails, float *avg_loops) {
    int ok = 0;
    float loops = 0;
    for (int i = 0; i < n; i++) {
        char p[PCAP];
        task_gen(task, level, p, PCAP);
        Solution s;
        if (adaptive) {
            infer_solve(M, kv, p, votes, &s);
        } else {
            infer_answer(M, kv, p, M->c.loops, 0, s.ans, sizeof(s.ans), NULL);
            s.loops = M->c.loops;
        }
        loops += (float)s.loops;
        if (task_check(p, s.ans)) ok++;
        else if (fails) ring_push(fails, p);
    }
    if (avg_loops) *avg_loops = loops / (float)n;
    return (float)ok / (float)n;
}

static int star_frontier(Model *M, KV *kv, int task, int level) {
    int kept = 0;
    for (int i = 0; i < STAR_N; i++) {
        char p[PCAP], a[64];
        task_gen(task, level, p, PCAP);
        for (int attempt = 0; attempt < 2; attempt++) {
            infer_answer(M, kv, p, M->c.loops, 0.8f, a, sizeof(a), NULL);
            if (task_check(p, a)) { ring_push(&g_self, p); kept++; break; }
        }
    }
    return kept;
}

static void curriculum_step(Model *M, KV *kv) {
    int *L = levels(M);
    printf("  curriculum:");
    for (int t = 0; t < T_COUNT; t++) {
        float acc = eval_level(M, kv, t, L[t], EVAL_N, 0, 1, &g_hard, NULL);
        int star = 0;
        if (L[t] < TASKS[t].train_max) star = star_frontier(M, kv, t, L[t] + 1);
        printf(" %s@%d=%.0f%%", TASKS[t].name, L[t], acc * 100);
        if (star) printf("(+%d)", star);
        if (acc >= MASTERY && L[t] < TASKS[t].train_max) L[t]++;
    }
    printf("  hard=%d self=%d\n", g_hard.n, g_self.n);
}

void train_run(Model *M, const TrainOpts *o) {
    Work *w = work_new(M, o->B, o->T);
    KV *kv = kv_new(M);
    if (!w) { fprintf(stderr, "error: T > seq\n"); return; }
    levels(M);
    double t0 = omp_get_wtime(), lsum = 0;
    int lcnt = 0;
    long long tokens = 0;
    for (int s = 1; s <= o->steps; s++) {
        for (int b = 0; b < o->B; b++)
            fill_row(M, o, w->tok + (size_t)b * o->T, w->tgt + (size_t)b * o->T);
        /* random depth: the shared blocks must work at any loop count */
        int loops = M->c.loops;
        if (loops > 1 && rng_uniform() < 0.3f) loops = 1 + rng_int(M->c.loops);
        model_zero_grad(M);
        float loss = model_forward(M, w, loops);
        model_backward(M, w);
        model_clip(M, 1.0f);
        float warm = fminf(1.0f, (float)s / 200.0f);
        float cosf_ = 0.1f + 0.45f * (1.0f + cosf(3.14159265f * (float)s / (float)o->steps));
        model_adamw(M, o->lr * warm * cosf_, o->wd);
        lsum += loss; lcnt++;
        tokens += w->N;
        if (s % o->log_every == 0) {
            double dt = omp_get_wtime() - t0;
            printf("step %5d loss %.4f  %.0f tok/s  %.0fs\n", s, lsum / lcnt,
                   (double)tokens / dt, dt);
            lsum = 0; lcnt = 0;
            fflush(stdout);
        }
        if (s % o->eval_every == 0 || s == o->steps) {
            curriculum_step(M, kv);
            if (o->ckpt) model_save(M, o->ckpt);
            fflush(stdout);
        }
    }
    kv_free(kv);
    work_free(w);
}

float eval_report(Model *M, int n, int votes, int extra) {
    KV *kv = kv_new(M);
    double tot = 0;
    int cells = 0;
    printf("%-6s", "task");
    for (int l = 1; l <= 10; l++) printf("  L%-4d", l);
    printf("   (* = beyond training ceiling)\n");
    for (int t = 0; t < T_COUNT; t++) {
        printf("%-6s", TASKS[t].name);
        int maxl = TASKS[t].train_max + extra;
        if (maxl > 10) maxl = 10;
        for (int l = 1; l <= maxl; l++) {
            float acc = eval_level(M, kv, t, l, n, 1, votes, NULL, NULL);
            printf("  %3.0f%%%c", acc * 100, l > TASKS[t].train_max ? '*' : ' ');
            if (l <= TASKS[t].train_max) { tot += acc; cells++; }
        }
        printf("\n");
        fflush(stdout);
    }
    kv_free(kv);
    float mean = cells ? (float)(tot / cells) : 0;
    printf("mean accuracy within training ceiling: %.1f%%\n", mean * 100);
    return mean;
}
