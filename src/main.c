#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "attention.h"
#include "config.h"
#include "neural_mem.h"
#include "tensor.h"
#include "tokenizer.h"
#include "train.h"

static unsigned char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    unsigned char *buf = malloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = '\0';
    *len_out = rd;
    return buf;
}

static void print_info(void) {
    Model probe;
    memset(&probe, 0, sizeof(probe));
    model_init(&probe);
    printf("MemCore\n");
    printf("  vocab=%d d_model=%d heads=%d layers=%d seq=%d ff=%d mem_hidden=%d\n",
           VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN, FF_HIDDEN,
           MEM_HIDDEN);
    printf("  core params: %lld (%.2f MB int8 / %.2f MB fp32)\n",
           model_param_count(&probe),
           (float)model_param_count(&probe) / 1048576.0f,
           (float)model_param_count(&probe) * 4.0f / 1048576.0f);
    model_free(&probe);
}

static int cmd_train(const char *file, const char *prefix, int max_windows,
                     float lr, int use_replay, int use_mem) {
    size_t len = 0;
    unsigned char *text = read_file(file, &len);
    if (!text) {
        fprintf(stderr, "error: cannot read %s\n", file);
        return 1;
    }
    Session *s = sess_new(42);
    s->core_lr = lr;
    s->use_replay = use_replay;
    s->use_mem = use_mem;
    char core_path[512], mem_path[512];
    snprintf(core_path, sizeof(core_path), "%s.core", prefix);
    snprintf(mem_path, sizeof(mem_path), "%s.mem", prefix);
    if (sess_load(s, core_path, mem_path) == 0)
        printf("[resumed from %s]\n", prefix);

    printf("training on %s (%zu bytes)\n", file, len);
    float loss = sess_train_bytes(s, text, len, max_windows, 25);
    printf("final loss=%.4f ema=%.4f steps=%lld surprise=%.4f\n", loss,
           s->loss_ema, s->step, s->mem->last_surprise);

    if (sess_save(s, core_path, mem_path) == 0)
        printf("saved %s.core + %s.mem\n", prefix, prefix);
    sess_free(s);
    free(text);
    return 0;
}

static int cmd_gen(const char *prefix, const char *prompt, int n, float temp,
                   int use_mem) {
    Session *s = sess_new(42);
    char core_path[512], mem_path[512];
    snprintf(core_path, sizeof(core_path), "%s.core", prefix);
    snprintf(mem_path, sizeof(mem_path), "%s.mem", prefix);
    if (sess_load(s, core_path, mem_path) != 0)
        printf("[no checkpoint found, using random init]\n");
    s->temperature = temp;

    unsigned char out[1024];
    sess_generate(s, (const unsigned char *)prompt, strlen(prompt), n, out,
                  sizeof(out) - 1, use_mem);
    out[sizeof(out) - 1] = '\0';
    printf("%.*s\n", n, (char *)out);
    sess_free(s);
    return 0;
}

static void chat_help(void) {
    printf("commands: :exit | :save | :temp <f> | anything else = learn+reply\n");
}

static int cmd_chat(const char *prefix) {
    Session *s = sess_new(42);
    char core_path[512], mem_path[512];
    snprintf(core_path, sizeof(core_path), "%s.core", prefix);
    snprintf(mem_path, sizeof(mem_path), "%s.mem", prefix);
    if (sess_load(s, core_path, mem_path) == 0)
        printf("[loaded %s]\n", prefix);
    else
        printf("[fresh model - it will learn everything you type]\n");
    chat_help();

    char line[4096];
    for (;;) {
        printf("\nyou> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        size_t ll = strcspn(line, "\r\n");
        line[ll] = '\0';
        if (ll == 0) continue;
        if (strcmp(line, ":exit") == 0) break;
        if (strcmp(line, ":save") == 0) {
            if (sess_save(s, core_path, mem_path) == 0)
                printf("[saved]\n");
            continue;
        }
        if (strncmp(line, ":temp ", 6) == 0) {
            s->temperature = (float)atof(line + 6);
            printf("[temperature=%.2f]\n", s->temperature);
            continue;
        }

        int window[SEQ_LEN + 1];
        if (ll >= SEQ_LEN + 1) {
            tok_encode((const unsigned char *)line, SEQ_LEN + 1, window,
                       SEQ_LEN + 1);
        } else {
            tok_encode((const unsigned char *)line, ll, window, SEQ_LEN + 1);
            for (size_t i = ll; i < SEQ_LEN + 1; i++) window[i] = ' ';
        }
        sess_train_window(s, window);

        unsigned char reply[256];
        sess_generate(s, (const unsigned char *)line, ll, 80, reply,
                      sizeof(reply) - 1, s->use_mem);
        printf("memcore> %s\n", (char *)reply);
        printf("[learned | loss_ema=%.4f]", s->loss_ema);
    }
    if (sess_save(s, core_path, mem_path) == 0)
        printf("\n[saved session to %s.core/.mem]\n", prefix);
    sess_free(s);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_info();
        printf("usage:\n"
               "  memcore train <file> [-o prefix] [-n max_windows]\n"
               "  memcore gen \"<prompt>\" [n] [-o prefix]\n"
               "  memcore chat [-o prefix]\n"
               "  memcore info\n");
        return 0;
    }

    rng_seed(20260825ULL);

    if (strcmp(argv[1], "info") == 0) return print_info(), 0;

    if (strcmp(argv[1], "train") == 0) {
        if (argc < 3) { fprintf(stderr, "need a file\n"); return 1; }
        const char *file = argv[2];
        const char *prefix = "model";
        int max_windows = 10000;
        float lr = 3e-4f;
        int use_replay = 1, use_mem = 1;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
                prefix = argv[++i];
            else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
                max_windows = atoi(argv[++i]);
            else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc)
                lr = (float)atof(argv[++i]);
            else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc)
                use_replay = atoi(argv[++i]);
            else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
                use_mem = atoi(argv[++i]);
        }
        return cmd_train(file, prefix, max_windows, lr, use_replay, use_mem);
    }

    if (strcmp(argv[1], "gen") == 0) {
        const char *prompt = argc > 2 ? argv[2] : "";
        int n = 200;
        const char *prefix = "model";
        float temp = 0.8f;
        int use_mem = 1;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
                prefix = argv[++i];
            else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc)
                temp = (float)atof(argv[++i]);
            else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)
                use_mem = atoi(argv[++i]);
            else
                n = atoi(argv[i]);
        }
        return cmd_gen(prefix, prompt, n, temp, use_mem);
    }

    if (strcmp(argv[1], "chat") == 0) {
        const char *prefix = argc > 3 ? argv[3] : "model";
        return cmd_chat(prefix);
    }

    fprintf(stderr, "unknown command: %s\n", argv[1]);
    return 1;
}
