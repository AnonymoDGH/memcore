#include "infer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

#define MAX_ANS 24
#define CONF_THRESH 0.9f

static void softmax_t(float *x, int V, float temp) {
    float t = temp > 0.01f ? temp : 1.0f, mx = x[0], s = 0;
    for (int v = 1; v < V; v++) if (x[v] > mx) mx = x[v];
    for (int v = 0; v < V; v++) { x[v] = expf((x[v] - mx) / t); s += x[v]; }
    for (int v = 0; v < V; v++) x[v] /= s;
}

static int pick(const float *p, int V, float temp) {
    if (temp <= 0.01f) {
        int b = 0;
        for (int v = 1; v < V; v++) if (p[v] > p[b]) b = v;
        return b;
    }
    float r = rng_uniform(), c = 0;
    for (int v = 0; v < V; v++) { c += p[v]; if (r <= c) return v; }
    return V - 1;
}

void infer_answer(const Model *M, KV *kv, const char *prompt, int loops,
                  float temp, char *ans, size_t cap, float *conf) {
    int V = M->c.vocab, pos = 0;
    float *lg = malloc(sizeof(float) * V), *pr = malloc(sizeof(float) * V);
    size_t pl = strlen(prompt), k = 0;
    float cmin = 1.0f;
    model_step(M, kv, '\n', pos++, loops, NULL, NULL);
    for (size_t i = 0; i < pl && pos < M->c.seq; i++)
        model_step(M, kv, (unsigned char)prompt[i], pos++, loops,
                   i + 1 == pl ? lg : NULL, NULL);
    while (k + 1 < cap && k < MAX_ANS && pos < M->c.seq) {
        memcpy(pr, lg, sizeof(float) * V);
        softmax_t(pr, V, 1.0f);
        int tk;
        if (temp > 0.01f) {
            softmax_t(lg, V, temp);
            tk = pick(lg, V, temp);
        } else {
            tk = pick(pr, V, 0);
        }
        if (pr[tk] < cmin) cmin = pr[tk];
        if (tk == '\n') break;
        ans[k++] = (char)tk;
        model_step(M, kv, tk, pos++, loops, lg, NULL);
    }
    ans[k] = '\0';
    if (conf) *conf = cmin;
    free(lg);
    free(pr);
}

void infer_solve(const Model *M, KV *kv, const char *prompt, int votes,
                 Solution *out) {
    int maxl = M->c.loops;
    infer_answer(M, kv, prompt, 1, 0, out->ans, sizeof(out->ans), &out->conf);
    out->loops = 1;
    if (maxl > 1 && out->conf < CONF_THRESH) {
        infer_answer(M, kv, prompt, maxl, 0, out->ans, sizeof(out->ans),
                     &out->conf);
        out->loops = maxl;
    }
    out->agree = 1.0f;
    if (votes <= 1) return;

    char cand[33][64];
    int cnt[33], nc = 0;
    strcpy(cand[nc], out->ans);
    cnt[nc++] = 1;
    if (votes > 32) votes = 32;
    for (int i = 1; i < votes; i++) {
        char a[64];
        infer_answer(M, kv, prompt, maxl, 0.7f, a, sizeof(a), NULL);
        int j = 0;
        while (j < nc && strcmp(cand[j], a) != 0) j++;
        if (j == nc) { strcpy(cand[nc], a); cnt[nc++] = 0; }
        cnt[j]++;
    }
    int best = 0;
    for (int j = 1; j < nc; j++) if (cnt[j] > cnt[best]) best = j;
    strcpy(out->ans, cand[best]);
    out->agree = (float)cnt[best] / (float)votes;
    out->loops = maxl;
}

size_t infer_generate(const Model *M, KV *kv, const KMem *mem,
                      const unsigned char *prompt, size_t plen, int n,
                      float temp, unsigned char *out, size_t cap) {
    int V = M->c.vocab, seq = M->c.seq, loops = M->c.loops;
    size_t tot = plen + (size_t)n + 1;
    int *ctx = malloc(sizeof(int) * tot);
    size_t len = 0, w = 0;
    ctx[len++] = '\n';
    for (size_t i = 0; i < plen; i++) ctx[len++] = prompt[i];
    float *lg = malloc(sizeof(float) * V), *h = malloc(sizeof(float) * M->c.d);
    int pos = 0;
    size_t start = len > (size_t)seq ? len - (size_t)seq / 2 : 0;
    for (size_t i = start; i < len; i++)
        model_step(M, kv, ctx[i], pos++, loops, lg, h);
    for (int g = 0; g < n && w + 1 < cap; g++) {
        softmax_t(lg, V, 1.0f);
        if (mem) kmem_mix(mem, h, lg, V, 0.9f);
        if (temp > 0.01f && fabsf(temp - 1.0f) > 1e-3f) {
            float s = 0;
            for (int v = 0; v < V; v++) { lg[v] = powf(lg[v], 1.0f / temp); s += lg[v]; }
            for (int v = 0; v < V; v++) lg[v] /= s;
        }
        int tk = pick(lg, V, temp);
        out[w++] = (unsigned char)tk;
        ctx[len++] = tk;
        if (pos >= seq) { /* slide: re-prefill the last half window */
            pos = 0;
            for (size_t i = len - (size_t)seq / 2; i < len; i++)
                model_step(M, kv, ctx[i], pos++, loops, lg, h);
        } else {
            model_step(M, kv, tk, pos++, loops, lg, h);
        }
    }
    out[w] = '\0';
    free(ctx); free(lg); free(h);
    return w;
}
