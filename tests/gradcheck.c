/* Numeric vs analytic gradients for every parameter tensor. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "tensor.h"

static float loss_at(Model *M, Work *w) { return model_forward(M, w, 2); }

int main(void) {
    Config c = {32, 16, 2, 2, 2, 24, 8};
    Model *M = model_new(c, 7);
    for (size_t i = 0; i < M->n_params; i++) M->p[i] += rng_normal() * 0.05f;
    Work *w = work_new(M, 2, 8);
    rng_seed(3);
    for (int n = 0; n < w->N; n++) {
        w->tok[n] = rng_int(32);
        w->tgt[n] = (n % 3 == 0) ? -1 : rng_int(32);
    }
    model_zero_grad(M);
    loss_at(M, w);
    model_backward(M, w);

    struct { const char *name; size_t off, len; } seg[] = {
        {"emb", M->o_emb, 32 * 16},   {"g1", M->o_g1[1], 16},
        {"qkv", M->o_qkv[0], 16 * 48}, {"wo", M->o_wo[1], 256},
        {"g2", M->o_g2[0], 16},       {"wg", M->o_wg[1], 16 * 24},
        {"wu", M->o_wu[0], 16 * 24},  {"wd", M->o_wd[1], 24 * 16},
        {"gf", M->o_gf, 16},
    };
    int fails = 0, total = 0;
    float eps = 1e-2f;
    for (size_t s = 0; s < sizeof(seg) / sizeof(seg[0]); s++) {
        float worst = 0;
        for (int k = 0; k < 6; k++) {
            size_t i = seg[s].off + (size_t)rng_int((int)seg[s].len);
            float orig = M->p[i];
            M->p[i] = orig + eps; float lp = loss_at(M, w);
            M->p[i] = orig - eps; float lm = loss_at(M, w);
            M->p[i] = orig;
            float num = (lp - lm) / (2 * eps), ana = M->g[i];
            float rel = fabsf(num - ana) / fmaxf(fabsf(num) + fabsf(ana), 1e-3f);
            if (rel > worst) worst = rel;
            total++;
            if (rel > 0.05f) fails++;
        }
        printf("%-4s max rel err %.4f\n", seg[s].name, worst);
    }
    printf("gradcheck: %d/%d ok\n", total - fails, total);

    /* KV-cache decoding must match the batched forward (row 0). */
    model_forward(M, w, 2);
    KV *kv = kv_new(M);
    float lg[32], worst = 0;
    for (int t = 0; t < 8; t++) {
        model_step(M, kv, w->tok[t], t, 2, lg, NULL);
        float mx = lg[0], sum = 0;
        for (int v = 1; v < 32; v++) if (lg[v] > mx) mx = lg[v];
        for (int v = 0; v < 32; v++) sum += expf(lg[v] - mx);
        for (int v = 0; v < 32; v++) {
            float e = fabsf(expf(lg[v] - mx) / sum - w->logits[t * 32 + v]);
            if (e > worst) worst = e;
        }
    }
    printf("kv-cache vs batch: max abs diff %.2e\n", worst);
    if (worst > 1e-4f) fails++;
    kv_free(kv);
    work_free(w);
    model_free(M);
    return fails ? 1 : 0;
}
