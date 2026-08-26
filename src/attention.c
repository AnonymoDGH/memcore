#include "attention.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GELU_C 0.7978845608f
#define ADAM_B1 0.9f
#define ADAM_B2 0.999f
#define ADAM_EPS 1e-8f
#define MAX_PARAMS_USED (2 + N_LAYERS * 16 + 3)
#define MAX_PARAMS MAX_PARAMS_USED

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(GELU_C * (x + 0.044715f * x * x * x)));
}

static float gelu_grad(float x) {
    float u = GELU_C * (x + 0.044715f * x * x * x);
    float t = tanhf(u);
    float s = 1.0f / coshf(u);
    return 0.5f * (1.0f + t) +
           0.5f * x * (s * s) * GELU_C * (1.0f + 3.0f * 0.044715f * x * x);
}

static void linear_fwd(float *y, const float *x, const Mat *w, const Mat *b) {
    int in = w->rows, out = w->cols;
    for (int o = 0; o < out; o++) y[o] = b ? b->data[o] : 0.0f;
    for (int i = 0; i < in; i++) {
        float xi = x[i];
        const float *wr = w->data + (size_t)i * out;
        for (int o = 0; o < out; o++) y[o] += xi * wr[o];
    }
}

static void linear_bwd(const float *x, const float *dy, Mat *dw, Mat *db,
                       float *dx, const Mat *w) {
    int in = w->rows, out = w->cols;
    for (int o = 0; o < out; o++) db->data[o] += dy[o];
    for (int i = 0; i < in; i++) {
        float xi = x[i];
        const float *wr = w->data + (size_t)i * out;
        float acc = 0.0f;
        for (int o = 0; o < out; o++) {
            dw->data[(size_t)i * out + o] += xi * dy[o];
            acc += wr[o] * dy[o];
        }
        if (dx) dx[i] += acc;
    }
}

static void ln_stats(float *mean_out, float *rstd_out, const float *in,
                     int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += in[i];
    mean /= (float)n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
    var /= (float)n;
    *mean_out = mean;
    *rstd_out = 1.0f / sqrtf(var + 1e-5f);
}

static void ln_bwd_accum(float *dx_in, const float *dout, const float *in,
                         const float *gamma, Mat *dg, Mat *db, float mean,
                         float rstd, int n) {
    float inv_n = 1.0f / (float)n;
    float sum_dxhat = 0.0f, sum_dxhat_xhat = 0.0f;
    for (int i = 0; i < n; i++) {
        float dxh = dout[i] * gamma[i];
        float xhat = (in[i] - mean) * rstd;
        dg->data[i] += dout[i] * xhat;
        db->data[i] += dout[i];
        sum_dxhat += dxh;
        sum_dxhat_xhat += dxh * xhat;
    }
    for (int i = 0; i < n; i++) {
        float dxh = dout[i] * gamma[i];
        float xhat = (in[i] - mean) * rstd;
        dx_in[i] +=
            rstd * (dxh - inv_n * sum_dxhat - inv_n * xhat * sum_dxhat_xhat);
    }
}

static void model_alloc_grads(Model *g) {
    g->tok_emb = mat_new(VOCAB_SIZE, D_MODEL);
    g->pos_emb = mat_new(SEQ_LEN, D_MODEL);
    for (int l = 0; l < N_LAYERS; l++) {
        g->ln1_g[l] = mat_new(1, D_MODEL);
        g->ln1_b[l] = mat_new(1, D_MODEL);
        g->wq[l] = mat_new(D_MODEL, D_MODEL);
        g->bq[l] = mat_new(1, D_MODEL);
        g->wk[l] = mat_new(D_MODEL, D_MODEL);
        g->bk[l] = mat_new(1, D_MODEL);
        g->wv[l] = mat_new(D_MODEL, D_MODEL);
        g->bv[l] = mat_new(1, D_MODEL);
        g->wo[l] = mat_new(D_MODEL, D_MODEL);
        g->ob[l] = mat_new(1, D_MODEL);
        g->ln2_g[l] = mat_new(1, D_MODEL);
        g->ln2_b[l] = mat_new(1, D_MODEL);
        g->fc1[l] = mat_new(D_MODEL, FF_HIDDEN);
        g->fb1[l] = mat_new(1, FF_HIDDEN);
        g->fc2[l] = mat_new(FF_HIDDEN, D_MODEL);
        g->fb2[l] = mat_new(1, D_MODEL);
    }
    g->lnf_g = mat_new(1, D_MODEL);
    g->lnf_b = mat_new(1, D_MODEL);
    g->wout = mat_new(D_MODEL, VOCAB_SIZE);
}

Trainer *trainer_new(void) {
    Trainer *tr = calloc(1, sizeof(Trainer));
    model_init(&tr->m);
    memset(&tr->g, 0, sizeof(Model));
    model_alloc_grads(&tr->g);
    memset(&tr->adam_m, 0, sizeof(Model));
    model_alloc_grads(&tr->adam_m);
    memset(&tr->adam_v, 0, sizeof(Model));
    model_alloc_grads(&tr->adam_v);
    act_alloc(&tr->act);
    return tr;
}

void trainer_free(Trainer *tr) {
    model_free(&tr->m);
    model_free(&tr->g);
    model_free(&tr->adam_m);
    model_free(&tr->adam_v);
    act_free(&tr->act);
    free(tr);
}

void act_alloc(Activations *a) {
    int S = SEQ_LEN, D = D_MODEL, F = FF_HIDDEN, V = VOCAB_SIZE;
    size_t SD = (size_t)S * D;
    a->seq = S;
    a->x_in = malloc(SD * sizeof(float));
    a->x_final = malloc(SD * sizeof(float));
    a->probs_final = malloc((size_t)S * V * sizeof(float));
    a->logits_core = malloc((size_t)S * V * sizeof(float));
    for (int l = 0; l <= N_LAYERS; l++)
        a->x_layer[l] = malloc(SD * sizeof(float));
    for (int l = 0; l < N_LAYERS; l++) {
        a->ln1[l] = malloc(SD * sizeof(float));
        a->q[l] = malloc(SD * sizeof(float));
        a->k[l] = malloc(SD * sizeof(float));
        a->v[l] = malloc(SD * sizeof(float));
        a->attn_cat[l] = malloc(SD * sizeof(float));
        a->attn_proj[l] = malloc(SD * sizeof(float));
        a->x_mid[l] = malloc(SD * sizeof(float));
        a->ln2[l] = malloc(SD * sizeof(float));
        a->ff_pre[l] = malloc((size_t)S * F * sizeof(float));
        a->ff_act[l] = malloc((size_t)S * F * sizeof(float));
        a->ff_out[l] = malloc(SD * sizeof(float));
        a->probs[l] = malloc((size_t)N_HEADS * S * S * sizeof(float));
        for (int j = 0; j < 2; j++) {
            a->ln_mean[l][j] = malloc(S * sizeof(float));
            a->ln_rstd[l][j] = malloc(S * sizeof(float));
        }
    }
    a->lnf_mean = malloc(S * sizeof(float));
    a->lnf_rstd = malloc(S * sizeof(float));
}

void act_free(Activations *a) {
    free(a->x_in);
    free(a->x_final);
    free(a->probs_final);
    free(a->logits_core);
    for (int l = 0; l <= N_LAYERS; l++) free(a->x_layer[l]);
    for (int l = 0; l < N_LAYERS; l++) {
        free(a->ln1[l]);
        free(a->q[l]); free(a->k[l]); free(a->v[l]);
        free(a->attn_cat[l]); free(a->attn_proj[l]);
        free(a->x_mid[l]); free(a->ln2[l]);
        free(a->ff_pre[l]); free(a->ff_act[l]); free(a->ff_out[l]);
        free(a->probs[l]);
        for (int j = 0; j < 2; j++) {
            free(a->ln_mean[l][j]);
            free(a->ln_rstd[l][j]);
        }
    }
    free(a->lnf_mean);
    free(a->lnf_rstd);
    memset(a, 0, sizeof(*a));
}

float model_forward(Trainer *tr, const int *tokens, const int *targets,
                    const float *mem_logits) {
    Model *m = &tr->m;
    Activations *a = &tr->act;
    int S = SEQ_LEN, D = D_MODEL, F = FF_HIDDEN, V = VOCAB_SIZE;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    for (int t = 0; t < S; t++)
        for (int d = 0; d < D; d++)
            a->x_in[t * D + d] = m->tok_emb->data[tokens[t] * D + d] +
                                 m->pos_emb->data[t * D + d];

    memcpy(a->x_layer[0], a->x_in, sizeof(float) * S * D);

    for (int l = 0; l < N_LAYERS; l++) {
        float *x = a->x_layer[l];
        for (int t = 0; t < S; t++) {
            ln_stats(&a->ln_mean[l][0][t], &a->ln_rstd[l][0][t], x + t * D, D);
            layer_norm(a->ln1[l] + t * D, x + t * D, m->ln1_g[l]->data,
                       m->ln1_b[l]->data, D);
        }
        for (int t = 0; t < S; t++) {
            linear_fwd(a->q[l] + t * D, a->ln1[l] + t * D, m->wq[l], m->bq[l]);
            linear_fwd(a->k[l] + t * D, a->ln1[l] + t * D, m->wk[l], m->bk[l]);
            linear_fwd(a->v[l] + t * D, a->ln1[l] + t * D, m->wv[l], m->bv[l]);
        }
        for (int h = 0; h < N_HEADS; h++) {
            int off = h * HEAD_DIM;
            for (int i = 0; i < S; i++) {
                float scores[SEQ_LEN];
                for (int j = 0; j <= i; j++) {
                    float dot = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++)
                        dot += a->q[l][i * D + off + d] *
                               a->k[l][j * D + off + d];
                    scores[j] = dot * scale;
                }
                softmax_inplace(scores, i + 1);
                float *prow = a->probs[l] + ((size_t)h * S + i) * S;
                memset(prow, 0, sizeof(float) * S);
                for (int j = 0; j <= i; j++) prow[j] = scores[j];
                for (int d = 0; d < HEAD_DIM; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j <= i; j++)
                        acc += scores[j] * a->v[l][j * D + off + d];
                    a->attn_cat[l][i * D + off + d] = acc;
                }
            }
        }
        for (int t = 0; t < S; t++)
            linear_fwd(a->attn_proj[l] + t * D, a->attn_cat[l] + t * D,
                       m->wo[l], m->ob[l]);
        for (int t = 0; t < S; t++)
            for (int d = 0; d < D; d++)
                a->x_mid[l][t * D + d] =
                    x[t * D + d] + a->attn_proj[l][t * D + d];

        for (int t = 0; t < S; t++) {
            ln_stats(&a->ln_mean[l][1][t], &a->ln_rstd[l][1][t],
                     a->x_mid[l] + t * D, D);
            layer_norm(a->ln2[l] + t * D, a->x_mid[l] + t * D,
                       m->ln2_g[l]->data, m->ln2_b[l]->data, D);
        }
        for (int t = 0; t < S; t++) {
            linear_fwd(a->ff_pre[l] + t * F, a->ln2[l] + t * D, m->fc1[l],
                       m->fb1[l]);
            for (int f = 0; f < F; f++)
                a->ff_act[l][t * F + f] = gelu(a->ff_pre[l][t * F + f]);
            linear_fwd(a->ff_out[l] + t * D, a->ff_act[l] + t * F, m->fc2[l],
                       m->fb2[l]);
        }
        memcpy(a->x_layer[l + 1], a->x_mid[l], sizeof(float) * S * D);
        vec_add(a->x_layer[l + 1], a->ff_out[l], S * D);
    }

    float *xfinal = a->x_layer[N_LAYERS];
    memcpy(a->x_final, xfinal, sizeof(float) * S * D);
    for (int t = 0; t < S; t++) {
        ln_stats(&a->lnf_mean[t], &a->lnf_rstd[t], xfinal + t * D, D);
        layer_norm(a->lnf + t * D, xfinal + t * D, m->lnf_g->data,
                   m->lnf_b->data, D);
    }
    for (int t = 0; t < S; t++) {
        linear_fwd(a->logits_core + t * V, a->lnf + t * D, m->wout, NULL);
        if (mem_logits)
            vec_add(a->logits_core + t * V, mem_logits + t * V, V);
    }
    float loss = 0.0f;
    for (int t = 0; t < S; t++) {
        softmax_inplace(a->logits_core + t * V, V);
        memcpy(a->probs_final + t * V, a->logits_core + t * V,
               sizeof(float) * V);
        loss += -logf(fmaxf(a->probs_final[t * V + targets[t]], 1e-10f));
    }
    return loss / (float)S;
}

void model_backward(Trainer *tr, const int *tokens, const int *targets) {
    Model *m = &tr->m;
    Model *g = &tr->g;
    Activations *a = &tr->act;
    int S = SEQ_LEN, D = D_MODEL, F = FF_HIDDEN, V = VOCAB_SIZE;
    float inv_S = 1.0f / (float)S;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    size_t SD = (size_t)S * D;
    float *dlogits = malloc((size_t)S * V * sizeof(float));
    float *d_xfinal = calloc(SD, sizeof(float));
    float *d_blk_out = malloc(SD * sizeof(float));
    float *dx_mid = malloc(SD * sizeof(float));
    float *dx_blk_in = calloc(SD, sizeof(float));
    float *dattn_cat = malloc(SD * sizeof(float));
    float *dln1 = calloc(SD, sizeof(float));
    float *dq = malloc(SD * sizeof(float));
    float *dk = malloc(SD * sizeof(float));
    float *dv_ = malloc(SD * sizeof(float));

    for (int t = 0; t < S; t++)
        for (int v = 0; v < V; v++)
            dlogits[(size_t)t * V + v] =
                (a->probs_final[(size_t)t * V + v] -
                 (v == targets[t] ? 1.0f : 0.0f)) * inv_S;

    memset(d_xfinal, 0, SD * sizeof(float));
    for (int t = 0; t < S; t++) {
        const float *dl = dlogits + (size_t)t * V;
        const float *lf = a->lnf + t * D;
        float *d_lnf = dq;
        for (int d = 0; d < D; d++) {
            const float *wr = m->wout->data + (size_t)d * V;
            float acc = 0.0f;
            for (int v = 0; v < V; v++) {
                g->wout->data[(size_t)d * V + v] += lf[d] * dl[v];
                acc += wr[v] * dl[v];
            }
            d_lnf[d] = acc;
        }
        ln_bwd_accum(d_xfinal + t * D, d_lnf, a->x_final + t * D,
                     m->lnf_g->data, g->lnf_g, g->lnf_b, a->lnf_mean[t],
                     a->lnf_rstd[t], D);
    }
    memcpy(d_blk_out, d_xfinal, SD * sizeof(float));

    for (int l = N_LAYERS - 1; l >= 0; l--) {
        memcpy(dx_mid, d_blk_out, SD * sizeof(float));
        memset(dx_blk_in, 0, SD * sizeof(float));

        for (int t = 0; t < S; t++) {
            float *dffo = d_blk_out + t * D;
            float dvec_f[FF_HIDDEN], dvec_d[D_MODEL];
            linear_bwd(a->ff_act[l] + t * F, dffo, g->fc2[l], g->fb2[l],
                       dvec_f, m->fc2[l]);
            for (int f = 0; f < F; f++)
                dvec_f[f] *= gelu_grad(a->ff_pre[l][t * F + f]);
            memset(dvec_d, 0, sizeof(dvec_d));
            linear_bwd(a->ln2[l] + t * D, dvec_f, g->fc1[l], g->fb1[l],
                       dvec_d, m->fc1[l]);
            ln_bwd_accum(dx_mid + t * D, dvec_d, a->x_mid[l] + t * D,
                         m->ln2_g[l]->data, g->ln2_g[l], g->ln2_b[l],
                         a->ln_mean[l][1][t], a->ln_rstd[l][1][t], D);
        }

        for (int t = 0; t < S; t++)
            linear_bwd(a->attn_cat[l] + t * D, dx_mid + t * D, g->wo[l],
                       g->ob[l], dattn_cat + t * D, m->wo[l]);

        memset(dq, 0, SD * sizeof(float));
        memset(dk, 0, SD * sizeof(float));
        memset(dv_, 0, SD * sizeof(float));
        memset(dln1, 0, SD * sizeof(float));
        for (int h = 0; h < N_HEADS; h++) {
            int off = h * HEAD_DIM;
            for (int i = 0; i < S; i++) {
                const float *prow =
                    a->probs[l] + ((size_t)h * S + i) * S;
                float dp[SEQ_LEN];
                for (int j = 0; j <= i; j++) {
                    float acc = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        float da = dattn_cat[i * D + off + d];
                        acc += da * a->v[l][j * D + off + d];
                        dv_[j * D + off + d] += prow[j] * da;
                    }
                    dp[j] = acc;
                }
                float dot_pd = 0.0f;
                for (int j = 0; j <= i; j++) dot_pd += prow[j] * dp[j];
                for (int j = 0; j <= i; j++) {
                    float ds = prow[j] * (dp[j] - dot_pd) * scale;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        dq[i * D + off + d] += ds * a->k[l][j * D + off + d];
                        dk[j * D + off + d] += ds * a->q[l][i * D + off + d];
                    }
                }
            }
        }
        for (int t = 0; t < S; t++) {
            linear_bwd(a->ln1[l] + t * D, dq + t * D, g->wq[l], g->bq[l],
                       dln1 + t * D, m->wq[l]);
            linear_bwd(a->ln1[l] + t * D, dk + t * D, g->wk[l], g->bk[l],
                       dln1 + t * D, m->wk[l]);
            linear_bwd(a->ln1[l] + t * D, dv_ + t * D, g->wv[l], g->bv[l],
                       dln1 + t * D, m->wv[l]);
        }

        for (int t = 0; t < S; t++) {
            ln_bwd_accum(dx_blk_in + t * D, dln1 + t * D,
                         a->x_layer[l] + t * D, m->ln1_g[l]->data,
                         g->ln1_g[l], g->ln1_b[l],
                         a->ln_mean[l][0][t], a->ln_rstd[l][0][t], D);
            vec_add(dx_blk_in + t * D, dx_mid + t * D, D);
        }
        memcpy(d_blk_out, dx_blk_in, SD * sizeof(float));
    }

    for (int t = 0; t < S; t++)
        for (int d = 0; d < D; d++) {
            g->tok_emb->data[(size_t)tokens[t] * D + d] +=
                d_blk_out[t * D + d];
            g->pos_emb->data[t * D + d] += d_blk_out[t * D + d];
        }

    free(dlogits); free(d_xfinal); free(d_blk_out); free(dx_mid);
    free(dx_blk_in); free(dattn_cat); free(dln1);
    free(dq); free(dk); free(dv_);
}

static void collect(Model *s, Mat **list, int *n) {
    list[(*n)++] = s->tok_emb;
    list[(*n)++] = s->pos_emb;
    for (int l = 0; l < N_LAYERS; l++) {
        list[(*n)++] = s->ln1_g[l]; list[(*n)++] = s->ln1_b[l];
        list[(*n)++] = s->wq[l]; list[(*n)++] = s->bq[l];
        list[(*n)++] = s->wk[l]; list[(*n)++] = s->bk[l];
        list[(*n)++] = s->wv[l]; list[(*n)++] = s->bv[l];
        list[(*n)++] = s->wo[l]; list[(*n)++] = s->ob[l];
        list[(*n)++] = s->ln2_g[l]; list[(*n)++] = s->ln2_b[l];
        list[(*n)++] = s->fc1[l]; list[(*n)++] = s->fb1[l];
        list[(*n)++] = s->fc2[l]; list[(*n)++] = s->fb2[l];
    }
    list[(*n)++] = s->lnf_g;
    list[(*n)++] = s->lnf_b;
    list[(*n)++] = s->wout;
}

long long model_param_count(const Model *mo) {
    Mat *lst[MAX_PARAMS];
    int n = 0;
    collect((Model *)mo, lst, &n);
    long long total = 0;
    for (int i = 0; i < n; i++)
        total += (long long)lst[i]->rows * lst[i]->cols;
    return total;
}

#define CORE_MAGIC 0x4D434F52u

int model_save(const Model *mo, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    unsigned magic = CORE_MAGIC, count = MAX_PARAMS_USED;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&count, sizeof(count), 1, f);
    Mat *lst[MAX_PARAMS];
    int n = 0;
    collect((Model *)mo, lst, &n);
    for (int i = 0; i < n; i++) {
        int rc[2] = {lst[i]->rows, lst[i]->cols};
        fwrite(rc, sizeof(int), 2, f);
        fwrite(lst[i]->data, sizeof(float), (size_t)rc[0] * rc[1], f);
    }
    fclose(f);
    return 0;
}

int model_load(Model *mo, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    unsigned magic = 0, count = 0;
    fread(&magic, sizeof(magic), 1, f);
    fread(&count, sizeof(count), 1, f);
    if (magic != CORE_MAGIC || count != MAX_PARAMS_USED) {
        fclose(f);
        return -1;
    }
    Mat *lst[MAX_PARAMS];
    int n = 0;
    collect(mo, lst, &n);
    for (int i = 0; i < n; i++) {
        int rc[2];
        if (fread(rc, sizeof(int), 2, f) != 2 ||
            fread(lst[i]->data, sizeof(float), (size_t)rc[0] * rc[1], f) !=
                (size_t)rc[0] * rc[1]) {
            fclose(f);
            return -1;
        }
        if (rc[0] != lst[i]->rows || rc[1] != lst[i]->cols) {
            fclose(f);
            return -1;
        }
    }
    fclose(f);
    return 0;
}

void model_zero_grads(Trainer *tr) {
    Mat *lst[MAX_PARAMS];
    int n = 0;
    collect(&tr->g, lst, &n);
    for (int i = 0; i < n; i++) mat_fill(lst[i], 0.0f);
}

void model_adam_step(Trainer *tr, float lr) {
    Mat *pm[MAX_PARAMS], *pg[MAX_PARAMS], *pmm[MAX_PARAMS], *pv[MAX_PARAMS];
    int n1 = 0, n2 = 0, n3 = 0, n4 = 0;
    collect(&tr->m, pm, &n1);
    collect(&tr->g, pg, &n2);
    collect(&tr->adam_m, pmm, &n3);
    collect(&tr->adam_v, pv, &n4);

    tr->t_step++;
    float bc1 = 1.0f - powf(ADAM_B1, (float)tr->t_step);
    float bc2 = 1.0f - powf(ADAM_B2, (float)tr->t_step);

    for (int i = 0; i < n1; i++) {
        size_t cnt = (size_t)pm[i]->rows * pm[i]->cols;
        float *w = pm[i]->data, *gr = pg[i]->data;
        float *mm = pmm[i]->data, *vv = pv[i]->data;
        for (size_t e = 0; e < cnt; e++) {
            mm[e] = ADAM_B1 * mm[e] + (1.0f - ADAM_B1) * gr[e];
            vv[e] = ADAM_B2 * vv[e] + (1.0f - ADAM_B2) * gr[e] * gr[e];
            float mh = mm[e] / bc1;
            float vh = vv[e] / bc2;
            w[e] -= lr * mh / (sqrtf(vh) + ADAM_EPS);
        }
    }
}
