#ifndef MEMCORE_ATTENTION_H
#define MEMCORE_ATTENTION_H

#include "config.h"
#include "tensor.h"

typedef struct {
    Mat *tok_emb;
    Mat *pos_emb;

    Mat *ln1_g[N_LAYERS], *ln1_b[N_LAYERS];
    Mat *wq[N_LAYERS], *bq[N_LAYERS];
    Mat *wk[N_LAYERS], *bk[N_LAYERS];
    Mat *wv[N_LAYERS], *bv[N_LAYERS];
    Mat *wo[N_LAYERS], *ob[N_LAYERS];
    Mat *ln2_g[N_LAYERS], *ln2_b[N_LAYERS];
    Mat *fc1[N_LAYERS], *fb1[N_LAYERS];
    Mat *fc2[N_LAYERS], *fb2[N_LAYERS];

    Mat *lnf_g, *lnf_b;
    Mat *wout;
} Model;

typedef struct {
    float *x_in;
    float *x_final;
    float *x_layer[N_LAYERS + 1];
    float *ln1[N_LAYERS];
    float *q[N_LAYERS], *k[N_LAYERS], *v[N_LAYERS];
    float *probs[N_LAYERS];
    float *attn_cat[N_LAYERS];
    float *attn_proj[N_LAYERS];
    float *x_mid[N_LAYERS];
    float *ln2[N_LAYERS];
    float *ff_pre[N_LAYERS];
    float *ff_act[N_LAYERS];
    float *ff_out[N_LAYERS];
    float *ln_mean[N_LAYERS][2], *ln_rstd[N_LAYERS][2];
    float *lnf;
    float *lnf_mean, *lnf_rstd;
    float *probs_final;
    float *logits_core;
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

void model_init(Model *mo);
void model_zero(Model *mo);
void model_free(Model *mo);
long long model_param_count(const Model *mo);
int model_save(const Model *mo, const char *path);
int model_load(Model *mo, const char *path);

void act_alloc(Activations *a);
void act_free(Activations *a);

float model_forward(Trainer *tr, const int *tokens, const int *targets,
                    const float *mem_logits);
void model_backward(Trainer *tr, const int *tokens, const int *targets);
void model_zero_grads(Trainer *tr);
void model_adam_step(Trainer *tr, float lr);

#endif
