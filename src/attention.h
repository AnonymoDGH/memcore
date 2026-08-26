#ifndef MEMCORE_ATTENTION_H
#define MEMCORE_ATTENTION_H

#include "config.h"
#include "tensor.h"

typedef struct {
    Mat *tok_emb;                 /* V x D, tied with output head */
    Mat *attn_norm_g[N_LAYERS];   /* rmsnorm gains */
    Mat *wq[N_LAYERS], *wk[N_LAYERS], *wv[N_LAYERS], *wo[N_LAYERS];
    Mat *ffn_norm_g[N_LAYERS];
    Mat *wg[N_LAYERS], *wu[N_LAYERS], *wd[N_LAYERS]; /* SwiGLU */
    Mat *final_norm_g;
} Model;

typedef struct {
    float *x_layer[N_LAYERS + 1]; /* residual stream S*D each */
    float *xn1[N_LAYERS];
    float *q[N_LAYERS], *k[N_LAYERS], *v[N_LAYERS];
    float *probs[N_LAYERS];
    float *att[N_LAYERS];         /* attention output before proj */
    float *proj[N_LAYERS];
    float *x_mid[N_LAYERS];
    float *xn2[N_LAYERS];
    float *gate_pre[N_LAYERS], *up_pre[N_LAYERS], *swig[N_LAYERS];
    float *down_out[N_LAYERS];
    float *rms1[N_LAYERS], *rms2[N_LAYERS];
    float *x_final, *xhat_final, *rms_f;
    float *probs_final;
    float rope_cos[SEQ_LEN][HEAD_DIM / 2];
    float rope_sin[SEQ_LEN][HEAD_DIM / 2];
    int seq;
} Activations;

typedef struct {
    Model m;
    Model g;
    Model adam_m;
    Model adam_v;
    Activations act;
    long long t_step;
} Trainer;

Trainer *trainer_new(void);
void trainer_free(Trainer *tr);

float model_forward(Trainer *tr, const int *tokens, const int *targets,
                    const float *mem_logits);
void model_backward(Trainer *tr, const int *tokens, const int *targets);
void model_zero_grads(Trainer *tr);
float model_clip_grads(Trainer *tr, float max_norm);
void model_adam_step(Trainer *tr, float lr);

long long model_param_count(const Model *mo);
int model_save(const Model *mo, const char *path);
int model_load(Model *mo, const char *path);

#endif
