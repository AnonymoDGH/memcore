#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "attention.h"
#include "tensor.h"

static int window[SEQ_LEN];
static int targets[SEQ_LEN];

static float eval_loss(Model *m) {
    Activations a;
    memset(&a, 0, sizeof(a));
    act_alloc(&a);
    Trainer tmp;
    memset(&tmp, 0, sizeof(tmp));
    tmp.m = *m;
    tmp.act = a;
    float loss = model_forward(&tmp, window, targets, NULL);
    act_free(&a);
    return loss;
}

int main(void) {
    rng_seed(7);
    for (int t = 0; t < SEQ_LEN; t++) {
        window[t] = (int)(rng_uniform() * VOCAB_SIZE);
        targets[t] = (t + 1 < SEQ_LEN) ? window[t + 1] : window[0];
    }

    Trainer *tr = trainer_new();
    model_zero_grads(tr);
    float l0 = model_forward(tr, window, targets, NULL);
    model_backward(tr, window, targets);

    struct { const char *name; Mat *m; Mat *g; } probes[] = {
        {"wout", tr->m.wout, tr->g.wout},
        {"tok_emb", tr->m.tok_emb, tr->g.tok_emb},
        {"pos_emb", tr->m.pos_emb, tr->g.pos_emb},
        {"wq0", tr->m.wq[0], tr->g.wq[0]},
        {"wo0", tr->m.wo[0], tr->g.wo[0]},
        {"fc1_0", tr->m.fc1[0], tr->g.fc1[0]},
        {"fc2_0", tr->m.fc2[0], tr->g.fc2[0]},
        {"ln1_g0", tr->m.ln1_g[0], tr->g.ln1_g[0]},
    };

    const float eps = 1e-3f;
    int fails = 0;
    printf("grad check (analytic vs numeric):\n");
    for (size_t p = 0; p < sizeof(probes) / sizeof(probes[0]); p++) {
        Mat *m = probes[p].m;
        Mat *g = probes[p].g;
        size_t n = (size_t)m->rows * m->cols;
        size_t idxs[3] = {0, n / 2, n - 1};
        for (int k = 0; k < 3; k++) {
            size_t i = idxs[k];
            float orig = m->data[i];
            m->data[i] = orig + eps;
            float lp = eval_loss(&tr->m);
            m->data[i] = orig - eps;
            float lm = eval_loss(&tr->m);
            m->data[i] = orig;
            float num = (lp - lm) / (2.0f * eps);
            float ana = g->data[i];
            float denom = fabsf(num) > fabsf(ana) ? fabsf(num) : fabsf(ana);
            float rel = denom > 1e-6f ? fabsf(num - ana) / denom
                                      : fabsf(num - ana);
            int ok = rel < 2e-2f || fabsf(num - ana) < 1.2e-3f;
            if (!ok) fails++;
            printf("  %-8s[%6zu] num=% .6f ana=% .6f rel=%.2e %s\n",
                   probes[p].name, i, num, ana, rel, ok ? "ok" : "FAIL");
        }
    }
    (void)l0;
    printf("%s\n", fails ? "GRADIENT CHECK FAILED" : "all gradients ok");
    trainer_free(tr);
    return fails != 0;
}
