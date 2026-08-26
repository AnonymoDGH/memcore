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
    Trainer tmp;
    memset(&tmp, 0, sizeof(tmp));
    tmp.m = *m;
    /* minimal standalone activations for eval */
    tmp.act.x_layer[0] = NULL;
    return 0.0f;
    (void)a;
}

int main(void) {
    rng_seed(7);
    Trainer *tr = trainer_new();
    for (int t = 0; t < SEQ_LEN; t++) {
        window[t] = (int)(rng_uniform() * VOCAB_SIZE);
        targets[t] = (t + 1 < SEQ_LEN) ? window[t + 1] : window[0];
    }
    model_zero_grads(tr);
    float l0 = model_forward(tr, window, targets, NULL);
    model_backward(tr, window, targets);

    { Mat *lst[] = {tr->g.tok_emb, tr->g.wq[0], tr->g.wk[0], tr->g.wv[0],
                    tr->g.wo[0], tr->g.wg[0], tr->g.wu[0], tr->g.wd[0],
                    tr->g.attn_norm_g[0], tr->g.ffn_norm_g[0]};
      const char *nm[] = {"emb","wq","wk","wv","wo","wg","wu","wd",
                          "a_ng","f_ng"};
      for (int i = 0; i < 10; i++) {
          size_t n = (size_t)lst[i]->rows * lst[i]->cols;
          float mx = 0; int bad = 0;
          for (size_t e = 0; e < n; e++) {
              if (!isfinite(lst[i]->data[e])) bad++;
              else if (fabsf(lst[i]->data[e]) > mx) mx = fabsf(lst[i]->data[e]);
          }
          printf("max|g %-6s| = %12.4e  nonfinite=%d\n", nm[i], mx, bad);
      } }

    struct { const char *name; Mat *m; Mat *g; } probes[] = {
        {"tok_emb", tr->m.tok_emb, tr->g.tok_emb},
        {"wq0", tr->m.wq[0], tr->g.wq[0]},
        {"wk0", tr->m.wk[0], tr->g.wk[0]},
        {"wo0", tr->m.wo[0], tr->g.wo[0]},
        {"wg0", tr->m.wg[0], tr->g.wg[0]},
        {"wu0", tr->m.wu[0], tr->g.wu[0]},
        {"wd0", tr->m.wd[0], tr->g.wd[0]},
        {"attn_ng0", tr->m.attn_norm_g[0], tr->g.attn_norm_g[0]},
        {"ffn_ng0", tr->m.ffn_norm_g[0], tr->g.ffn_norm_g[0]},
    };

    const float eps = 5e-3f;
    int fails = 0;
    printf("loss=%.4f\ngrad check:\n", l0);
    for (size_t p = 0; p < sizeof(probes) / sizeof(probes[0]); p++) {
        Mat *m = probes[p].m;
        Mat *gg = probes[p].g;
        size_t n = (size_t)m->rows * m->cols;
        size_t idxs[3] = {0, n / 2, n - 1};
        for (int k = 0; k < 3; k++) {
            size_t i = idxs[k];
            float orig = m->data[i];
            m->data[i] = orig + eps;
            float lp = model_forward(tr, window, targets, NULL);
            m->data[i] = orig - eps;
            float lm = model_forward(tr, window, targets, NULL);
            m->data[i] = orig;
            float num = (lp - lm) / (2.0f * eps);
            float ana = gg->data[i];
            float denom = fabsf(num) > fabsf(ana) ? fabsf(num) : fabsf(ana);
            float rel = denom > 1e-6f ? fabsf(num - ana) / denom
                                      : fabsf(num - ana);
            int ok = rel < 3e-2f || fabsf(num - ana) < 1.5e-3f;
            if (!ok) fails++;
            printf("  %-9s[%6zu] num=% .6f ana=% .6f rel=%.2e %s\n",
                   probes[p].name, i, num, ana, rel, ok ? "ok" : "FAIL");
        }
    }
    printf("%s\n", fails ? "GRADIENT CHECK FAILED" : "all gradients ok");
    trainer_free(tr);
    return fails != 0;
}
