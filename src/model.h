#ifndef MC_MODEL_H
#define MC_MODEL_H

#include <stddef.h>
#include <stdint.h>

#define MC_MAXL 16
#define MC_MAXT 512
#define MC_META 32

/* Decoder-only transformer: RoPE, RMSNorm, SwiGLU, tied embeddings.
   `layers` unique blocks are applied up to `loops` times (weight-shared
   depth); the number of loops can vary per call (adaptive compute). */
typedef struct {
    int vocab, d, heads, layers, loops, ff, seq;
} Config;

typedef struct {
    Config c;
    size_t n_params;
    size_t o_emb, o_gf;
    size_t o_g1[MC_MAXL], o_qkv[MC_MAXL], o_wo[MC_MAXL];
    size_t o_g2[MC_MAXL], o_wg[MC_MAXL], o_wu[MC_MAXL], o_wd[MC_MAXL];
    float *p, *g, *m, *v; /* params, grads, adam moments */
    long long step;
    int meta[MC_META]; /* persisted trainer state (curriculum levels) */
    float *rcos, *rsin; /* seq x hd/2 */
} Model;

typedef struct {
    int B, T, N, A, A_run, nvalid; /* A = max applications */
    int *tok, *tgt; /* N each; tgt < 0 = no loss */
    float *x;       /* (A+1) residual streams of N*d */
    float *xn1, *qkv, *att, *probs, *xmid, *xn2, *gp, *up, *hh, *r1, *r2;
    float *xnf, *rf, *logits;
    float *dx, *dxn, *dqkv, *datt, *dhh, *dgp, *dup;
} Work;

typedef struct {
    int A, seq, d, ff;
    float *k, *v; /* A x seq x d */
    float *x, *xn, *qkv, *att, *tmp, *gp, *up, *sc;
} KV;

Model *model_new(Config c, uint64_t seed);
void model_free(Model *M);
int model_save(const Model *M, const char *path);
Model *model_load(const char *path);
void model_print(const Model *M);

Work *work_new(const Model *M, int B, int T);
void work_free(Work *w);

float model_forward(Model *M, Work *w, int loops);
void model_backward(Model *M, Work *w);
void model_zero_grad(Model *M);
float model_clip(Model *M, float max_norm);
void model_adamw(Model *M, float lr, float wd);

KV *kv_new(const Model *M);
void kv_free(KV *kv);
/* one-token incremental forward; pos < seq. hidden (d floats) optional */
void model_step(const Model *M, KV *kv, int tok, int pos, int loops,
                float *logits, float *hidden);

#endif
