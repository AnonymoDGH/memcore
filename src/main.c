#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "infer.h"
#include "memory.h"
#include "model.h"
#include "tasks.h"
#include "tensor.h"
#include "train.h"

static unsigned char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    unsigned char *buf = malloc((size_t)sz + 1);
    *len = fread(buf, 1, (size_t)sz, f);
    buf[*len] = '\0';
    fclose(f);
    return buf;
}

static const char *arg_s(int argc, char **argv, const char *flag,
                         const char *def) {
    for (int i = 2; i + 1 < argc; i++)
        if (strcmp(argv[i], flag) == 0) return argv[i + 1];
    return def;
}

static int arg_i(int argc, char **argv, const char *flag, int def) {
    const char *s = arg_s(argc, argv, flag, NULL);
    return s ? atoi(s) : def;
}

static float arg_f(int argc, char **argv, const char *flag, float def) {
    const char *s = arg_s(argc, argv, flag, NULL);
    return s ? (float)atof(s) : def;
}

static Model *open_model(const char *path) {
    Model *M = model_load(path);
    if (!M) fprintf(stderr, "error: cannot load %s (run `memcore train` first)\n", path);
    return M;
}

static void usage(void) {
    printf("MemCore v3 - small models that learn a lot from little data\n\n"
           "  memcore train [-o model.mc] [-s steps] [-b batch] [-T len] [-l lr]\n"
           "                [--text file --mix 0.3] [--d 128 --heads 4 --layers 3\n"
           "                 --loops 2 --ff 384 --seq 128]\n"
           "  memcore eval  [-o model.mc] [-n per_level] [-k votes]\n"
           "  memcore solve \"123+456\" [-o model.mc] [-k votes]\n"
           "  memcore learn <file> [-o model.mc] [-m memory.knn]\n"
           "  memcore gen \"prompt\" [-o model.mc] [-m memory.knn] [-n bytes] [-t temp]\n"
           "  memcore chat  [-o model.mc] [-m memory.knn]\n"
           "  memcore info  [-o model.mc]\n\n"
           "solve accepts: 123+45  900-7  123*4  cmp 12 34  rev hello  sort dcba  count abca\n");
}

static void print_solution(const char *prompt, const Solution *s) {
    int t = task_of(prompt);
    char pretty[64], ref[64], refp[64];
    task_pretty(t, s->ans, pretty, sizeof(pretty));
    task_solve(prompt, ref, sizeof(ref));
    task_pretty(t, ref, refp, sizeof(refp));
    int ok = strcmp(ref, s->ans) == 0;
    printf("%s %s   [%s, depth %d, conf %.2f", prompt, pretty, TASKS[t].name,
           s->loops, s->conf);
    if (s->agree < 1.0f) printf(", vote agreement %.0f%%", s->agree * 100);
    printf("] %s", ok ? "OK" : "WRONG");
    if (!ok) printf(" (expected %s)", refp);
    if (s->agree < 0.5f) printf("  -- low agreement: unsure");
    printf("\n");
}

static int cmd_train(int argc, char **argv, const char *path) {
    Model *M = model_load(path);
    if (M) {
        printf("[resuming %s]\n", path);
    } else {
        Config c = {256,
                    arg_i(argc, argv, "--d", 128),
                    arg_i(argc, argv, "--heads", 4),
                    arg_i(argc, argv, "--layers", 3),
                    arg_i(argc, argv, "--loops", 2),
                    arg_i(argc, argv, "--ff", 384),
                    arg_i(argc, argv, "--seq", 128)};
        M = model_new(c, 1234);
        if (!M) { fprintf(stderr, "error: invalid config\n"); return 1; }
    }
    model_print(M);
    TrainOpts o = {0};
    o.steps = arg_i(argc, argv, "-s", 3000);
    o.B = arg_i(argc, argv, "-b", 32);
    o.T = arg_i(argc, argv, "-T", 96);
    o.lr = arg_f(argc, argv, "-l", 2e-3f);
    o.wd = arg_f(argc, argv, "--wd", 0.05f);
    o.eval_every = arg_i(argc, argv, "-e", 250);
    o.log_every = 50;
    o.ckpt = path;
    const char *tf = arg_s(argc, argv, "--text", NULL);
    unsigned char *text = NULL;
    if (tf) {
        text = read_file(tf, &o.text_len);
        if (!text) { fprintf(stderr, "error: cannot read %s\n", tf); return 1; }
        o.text = text;
        o.text_mix = arg_f(argc, argv, "--mix", 0.3f);
    }
    if (o.T > M->c.seq) o.T = M->c.seq;
    train_run(M, &o);
    printf("saved %s\n", path);
    free(text);
    model_free(M);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 0; }
    const char *cmd = argv[1];
    const char *path = arg_s(argc, argv, "-o", "model.mc");
    const char *mpath = arg_s(argc, argv, "-m", "memory.knn");
    rng_seed(20261009ULL);

    if (strcmp(cmd, "train") == 0) return cmd_train(argc, argv, path);

    Model *M = open_model(path);
    if (!M) return 1;
    int rc = 0;

    if (strcmp(cmd, "info") == 0) {
        model_print(M);
        printf("  curriculum levels:");
        for (int t = 0; t < T_COUNT; t++)
            printf(" %s=%d/%d", TASKS[t].name, M->meta[t], TASKS[t].train_max);
        printf("\n");
    } else if (strcmp(cmd, "eval") == 0) {
        eval_report(M, arg_i(argc, argv, "-n", 100), arg_i(argc, argv, "-k", 1),
                    2);
    } else if (strcmp(cmd, "solve") == 0 && argc > 2) {
        char prompt[128];
        if (task_parse(argv[2], prompt, sizeof(prompt)) < 0) {
            fprintf(stderr, "error: cannot parse '%s'\n", argv[2]);
            rc = 1;
        } else {
            KV *kv = kv_new(M);
            Solution s;
            infer_solve(M, kv, prompt, arg_i(argc, argv, "-k", 1), &s);
            print_solution(prompt, &s);
            kv_free(kv);
        }
    } else if (strcmp(cmd, "learn") == 0 && argc > 2) {
        size_t len;
        unsigned char *text = read_file(argv[2], &len);
        if (!text) { fprintf(stderr, "error: cannot read %s\n", argv[2]); rc = 1; }
        else {
            KMem *km = kmem_load(mpath);
            if (!km) km = kmem_new(M->c.d);
            KV *kv = kv_new(M);
            kmem_learn(km, M, kv, text, len);
            kmem_save(km, mpath);
            printf("memorized %zu bytes -> %s (%d entries)\n", len, mpath, km->n);
            kv_free(kv);
            kmem_free(km);
            free(text);
        }
    } else if (strcmp(cmd, "gen") == 0 && argc > 2) {
        KMem *km = kmem_load(mpath);
        KV *kv = kv_new(M);
        unsigned char out[4096];
        infer_generate(M, kv, km, (const unsigned char *)argv[2], strlen(argv[2]),
                       arg_i(argc, argv, "-n", 120), arg_f(argc, argv, "-t", 0.0f),
                       out, sizeof(out));
        printf("%s%s\n", argv[2], (char *)out);
        kv_free(kv);
        kmem_free(km);
    } else if (strcmp(cmd, "chat") == 0) {
        KMem *km = kmem_load(mpath);
        if (!km) km = kmem_new(M->c.d);
        KV *kv = kv_new(M);
        printf("tasks are solved (e.g. 123+45, rev hola); any other line is\n"
               "memorized instantly; end a line with '...' to ask for completion.\n"
               ":exit saves memory\n");
        char line[1024];
        for (;;) {
            printf("> ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            line[strcspn(line, "\r\n")] = '\0';
            if (!line[0]) continue;
            if (strcmp(line, ":exit") == 0) break;
            char prompt[128];
            size_t ll = strlen(line);
            if (task_parse(line, prompt, sizeof(prompt)) >= 0) {
                Solution s;
                infer_solve(M, kv, prompt, 5, &s);
                print_solution(prompt, &s);
            } else if (ll > 3 && strcmp(line + ll - 3, "...") == 0) {
                unsigned char out[512];
                line[ll - 3] = '\0';
                infer_generate(M, kv, km, (unsigned char *)line, ll - 3, 80, 0,
                               out, sizeof(out));
                printf("%s%s\n", line, (char *)out);
            } else {
                line[ll++] = '\n';
                line[ll] = '\0';
                kmem_learn(km, M, kv, (unsigned char *)line, ll);
                printf("[memorized, %d entries]\n", km->n);
            }
        }
        kmem_save(km, mpath);
        kv_free(kv);
        kmem_free(km);
    } else {
        usage();
        rc = 1;
    }
    model_free(M);
    return rc;
}
