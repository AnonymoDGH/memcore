#include "attention.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ADAM_B1 0.9f
#define ADAM_B2 0.999f
#define ADAM_EPS 1e-8f
#define MAX_PARAMS 64

static float silu(float x) { return x / (1.0f + expf(-x)); }

static float silu_grad(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f + x * (1.0f - s));
}

static void linear_fwd(float *y, const float *x, const Mat *w) {
    int in = w->rows, out = w->cols;
    for (int o = 0; o < out; o++) y[o] = 0.0f;
    for (int i = 0; i < in; i++) {
        float xi = x[i];
        const float *wr = w->data + (size_t)i * out;
        for (int o = 0; o < out; o++) y[o] += xi * wr[o];
    }
}

static void linear_bwd(const float *x, const float *dy, Mat *dw, float *dx,
                       const Mat *w) {
    int in = w->rows, out = w->cols;
    if (dx) memset(dx, 0, sizeof(float) * in);
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

/* RMSNorm forward: out = x/rms(x) * g ; stores rms */
static void rms_fwd(float *out, const float *in, const Mat *g, int n,
                    float *rms_out) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)in[i] * in[i];
    float rms = (float)sqrt(ss / n + 1e-6f);
    *rms_out = rms;
    float inv = 1.0f / rms;
    for (int i = 0; i < n; i++) out[i] = in[i] * inv * g->data[i];
}

/* RMSNorm backward: a_i = g_i*dy_i ; dx = (a - xhat*mean(a*xhat))/rms ;
   dg += dy*xhat */
static void rms_bwd(float *dx_in, const float *dy, const float *in,
                    const Mat *g, Mat *dg, float rms, int n) {
    float inv = 1.0f / rms;
    float sum_ax = 0.0f;
    for (int i = 0; i < n; i++) {
        float xhat = in[i] * inv;
        dg->data[i] += dy[i] * xhat;
        sum_ax += g->data[i] * dy[i] * xhat;
    }
    float mean_ax = sum_ax / (float)n;
    for (int i = 0; i < n; i++) {
        float xhat = in[i] * inv;
        dx_in[i] += (g->data[i] * dy[i] - xhat * mean_ax) * inv;
    }
}

static void rope_tables(Activations *a) {
    for (int t = 0; t < SEQ_LEN; t++)
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            float freq =
                powf(10000.0f, -2.0f * (float)i / (float)(HEAD_DIM / 2));
            float ang = (float)t * freq;
            a->rope_cos[t][i] = cosf(ang);
            a->rope_sin[t][i] = sinf(ang);
        }
}

/* rotate pairs (i, i+hd/2) by angle tables at position t */
static void rope_apply(float *v, int t, const Activations *a) {
    for (int d = 0; d < N_HEADS; d++) {
        int off = d * HEAD_DIM;
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            int j = off + i, k = off + i + HEAD_DIM / 2;
            float c = a->rope_cos[t][i], s = a->rope_sin[t][i];
            float vj = v[j], vk = v[k];
            v[j] = vj * c - vk * s;
            v[k] = vj * s + vk * c;
        }
    }
}

static void rope_backward(float *dv, int t, const Activations *a) {
    for (int d = 0; d < N_HEADS; d++) {
        int off = d * HEAD_DIM;
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            int j = off + i, k = off + i + HEAD_DIM / 2;
            float c = a->rope_cos[t][i], s = a->rope_sin[t][i];
            float dj = dv[j], dk = dv[k];
            dv[j] = dj * c + dk * s;
            dv[k] = dk * c - dj * s;
        }
    }
}

#define CORE_MAGIC 0x4D433032u

static int count_params(void) {
    return 2 /*emb,final*/ +
           N_LAYERS * (1 + 4 + 1 + 3); /* norm,qkvo,norm,swiglu */
}

static void collect(Model *s, Mat **list, int *n) {
    list[(*n)++] = s->tok_emb;
    for (int l = 0; l < N_LAYERS; l++) {
        list[(*n)++] = s->attn_norm_g[l];
        list[(*n)++] = s->wq[l];
        list[(*n)++] = s->wk[l];
        list[(*n)++] = s->wv[l];
        list[(*n)++] = s->wo[l];
        list[(*n)++] = s->ffn_norm_g[l];
        list[(*n)++] = s->wg[l];
        list[(*n)++] = s->wu[l];
        list[(*n)++] = s->wd[l];
    }
    list[(*n)++] = s->final_norm_g;
}

void model_zero(Model *mo) {
    Mat *lst[MAX_PARAMS];
    int n = 0;
    collect(mo, lst, &n);
    for (int i = 0; i < n; i++) mat_fill(lst[i], 0.0f);
}

static Model model_alloc_empty(int with_random) {
    Model mo;
    memset(&mo, 0, sizeof(mo));
    mo.tok_emb = mat_new(VOCAB_SIZE, D_MODEL);
    for (int l = 0; l < N_LAYERS; l++) {
        mo.attn_norm_g[l] = mat_new(1, D_MODEL);
        mo.wq[l] = mat_new(D_MODEL, D_MODEL);
        mo.wk[l] = mat_new(D_MODEL, D_MODEL);
        mo.wv[l] = mat_new(D_MODEL, D_MODEL);
        mo.wo[l] = mat_new(D_MODEL, D_MODEL);
        mo.ffn_norm_g[l] = mat_new(1, D_MODEL);
        mo.wg[l] = mat_new(D_MODEL, FF_HIDDEN);
        mo.wu[l] = mat_new(D_MODEL, FF_HIDDEN);
        mo.wd[l] = mat_new(FF_HIDDEN, D_MODEL);
    }
    mo.final_norm_g = mat_new(1, D_MODEL);

    if (!with_random) return mo;

    for (int l = 0; l < N_LAYERS; l++) {
        mat_fill(mo.attn_norm_g[l], 1.0f);
        mat_fill(mo.ffn_norm_g[l], 1.0f);
        float sa = 1.0f / sqrtf((float)D_MODEL);
        mat_randn(mo.wq[l], sa);
        mat_randn(mo.wk[l], sa);
        mat_randn(mo.wv[l], sa);
        mat_randn(mo.wo[l], sa * 0.5f);
        mat_randn(mo.wg[l], sa);
        mat_randn(mo.wu[l], sa);
        float sd = 1.0f / sqrtf((float)FF_HIDDEN);
        mat_randn(mo.wd[l], sd);
    }
    mat_fill(mo.final_norm_g, 1.0f);
    mat_randn(mo.tok_emb, 0.02f);
    return mo;
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

Trainer *trainer_new(void) {
    Trainer *tr = calloc(1, sizeof(Trainer));
    tr->m = model_alloc_empty(1);
    tr->g = model_alloc_empty(0);
    tr->adam_m = model_alloc_empty(0);
    tr->adam_v = model_alloc_empty(0);

    Activations *a = &tr->act;
    int S = SEQ_LEN, D = D_MODEL, F = FF_HIDDEN, V = VOCAB_SIZE;
    size_t SD = (size_t)S * D;
    for (int l = 0; l <= N_LAYERS; l++)
        a->x_layer[l] = malloc(SD * sizeof(float));
    for (int l = 0; l < N_LAYERS; l++) {
        a->xn1[l] = malloc(SD * sizeof(float));
        a->q[l] = malloc(SD * sizeof(float));
        a->k[l] = malloc(SD * sizeof(float));
        a->v[l] = malloc(SD * sizeof(float));
        a->probs[l] = malloc((size_t)N_HEADS * S * S * sizeof(float));
        a->att[l] = malloc(SD * sizeof(float));
        a->proj[l] = malloc(SD * sizeof(float));
        a->x_mid[l] = malloc(SD * sizeof(float));
        a->xn2[l] = malloc(SD * sizeof(float));
        a->gate_pre[l] = malloc((size_t)S * F * sizeof(float));
        a->up_pre[l] = malloc((size_t)S * F * sizeof(float));
        a->swig[l] = malloc((size_t)S * F * sizeof(float));
        a->down_out[l] = malloc(SD * sizeof(float));
        a->rms1[l] = malloc(S * sizeof(float));
        a->rms2[l] = malloc(S * sizeof(float));
    }
    a->x_final = malloc(SD * sizeof(float));
    a->xhat_final = malloc(SD * sizeof(float));
    a->rms_f = malloc(S * sizeof(float));
    a->probs_final = malloc((size_t)S * V * sizeof(float));
    a->seq = S;
    rope_tables(a);
    return tr;
}

void trainer_free(Trainer *tr) {
    Mat *lst[MAX_PARAMS];
    Model *ms[4] = {&tr->m, &tr->g, &tr->adam_m, &tr->adam_v};
    for (int k = 0; k < 4; k++) {
        int n = 0;
        collect(ms[k], lst, &n);
        for (int i = 0; i < n; i++) mat_free(lst[i]);
    }
    Activations *a = &tr->act;
    for (int l = 0; l <= N_LAYERS; l++) free(a->x_layer[l]);
    for (int l = 0; l < N_LAYERS; l++) {
        free(a->xn1[l]); free(a->q[l]); free(a->k[l]); free(a->v[l]);
        free(a->probs[l]); free(a->att[l]); free(a->proj[l]);
        free(a->x_mid[l]); free(a->xn2[l]);
        free(a->gate_pre[l]); free(a->up_pre[l]); free(a->swig[l]);
        free(a->down_out[l]); free(a->rms1[l]); free(a->rms2[l]);
    }
    free(a->x_final); free(a->xhat_final); free(a->rms_f);
    free(a->probs_final);
    free(tr);
}

float model_forward(Trainer *tr, const int *tokens, const int *targets,
                    const float *mem_logits) {
    Model *m = &tr->m;
    Activations *a = &tr->act;
    int S = SEQ_LEN, D = D_MODEL, F = FF_HIDDEN, V = VOCAB_SIZE;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

#pragma omp parallel for schedule(static)
    for (int t = 0; t < S; t++)
        memcpy(a->x_layer[0] + t * D, m->tok_emb->data + tokens[t] * D,
               sizeof(float) * D);

    for (int l = 0; l < N_LAYERS; l++) {
        const float *x = a->x_layer[l];
#pragma omp parallel for schedule(static)
        for (int t = 0; t < S; t++)
            rms_fwd(a->xn1[l] + t * D, x + t * D, m->attn_norm_g[l], D,
                    &a->rms1[l][t]);

#pragma omp parallel for schedule(static)
                for (int t = 0; t < S; t++) {
            linear_fwd(a->q[l] + t * D, a->xn1[l] + t * D, m->wq[l]);
            linear_fwd(a->k[l] + t * D, a->xn1[l] + t * D, m->wk[l]);
            linear_fwd(a->v[l] + t * D, a->xn1[l] + t * D, m->wv[l]);
            rope_apply(a->q[l] + t * D, t, a);
            rope_apply(a->k[l] + t * D, t, a);
        }

#pragma omp parallel for collapse(2) schedule(static)
        for (int h = 0; h < N_HEADS; h++) {
            for (int i = 0; i < S; i++) {
                int off = h * HEAD_DIM;
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
                    a->att[l][i * D + off + d] = acc;
                }
            }
        }

        for (int t = 0; t < S; t++)
            linear_fwd(a->proj[l] + t * D, a->att[l] + t * D, m->wo[l]);
        memcpy(a->x_mid[l], x, sizeof(float) * S * D);
        vec_add(a->x_mid[l], a->proj[l], S * D);

#pragma omp parallel for schedule(static)
        for (int t = 0; t < S; t++)
            rms_fwd(a->xn2[l] + t * D, a->x_mid[l] + t * D,
                    m->ffn_norm_g[l], D, &a->rms2[l][t]);

#pragma omp parallel for schedule(static)
        for (int t = 0; t < S; t++) {
            float *gp = a->gate_pre[l] + t * F;
            float *up = a->up_pre[l] + t * F;
            float *sw = a->swig[l] + t * F;
            linear_fwd(gp, a->xn2[l] + t * D, m->wg[l]);
            linear_fwd(up, a->xn2[l] + t * D, m->wu[l]);
            for (int f = 0; f < F; f++) sw[f] = silu(gp[f]) * up[f];
            linear_fwd(a->down_out[l] + t * D, sw, m->wd[l]);
        }

        float *xo = a->x_layer[l + 1];
        memcpy(xo, a->x_mid[l], sizeof(float) * S * D);
        vec_add(xo, a->down_out[l], S * D);
    }

    float *xf = a->x_layer[N_LAYERS];
    memcpy(a->x_final, xf, sizeof(float) * S * D);
    for (int t = 0; t < S; t++) {
        rms_fwd(a->xhat_final + t * D, a->x_final + t * D, m->final_norm_g,
                D, &a->rms_f[t]);
    }
    const Mat *E = m->tok_emb; /* tied head: logits = xhat @ E^T */
#pragma omp parallel for schedule(static)
    for (int t = 0; t < S; t++) {
        const float *h = a->xhat_final + t * D;
        float tmp[VOCAB_SIZE];
        for (int v = 0; v < V; v++) {
            const float *er = E->data + (size_t)v * D;
            float dot = 0.0f;
            for (int d = 0; d < D; d++) dot += h[d] * er[d];
            tmp[v] = dot > 60.0f ? 60.0f : (dot < -60.0f ? -60.0f : dot);
        }
        if (mem_logits) vec_add(tmp, mem_logits + t * V, V);
        softmax_inplace(tmp, V);
        memcpy(a->probs_final + t * V, tmp, sizeof(float) * V);
    }

    float loss = 0.0f;
    for (int t = 0; t < S; t++)
        loss += -logf(fmaxf(
            a->probs_final[(size_t)t * V + targets[t]], 1e-10f));
    if (getenv("MEMCORE_DEBUG")) {
        float m1 = 0;
        for (int l = 0; l <= N_LAYERS; l++)
            for (size_t e = 0; e < (size_t)S * D; e++) {
                float av = fabsf(a->x_layer[l][e]);
                if (isfinite(av) && av > m1) m1 = av;
            }
        fprintf(stderr, "[dbg] max|x_layer|=%.3e loss=%.3f\n", m1,
                loss / S);
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
    float *dxhat = calloc(SD, sizeof(float));
    float *dres = malloc(SD * sizeof(float));
    float *dx_mid = malloc(SD * sizeof(float));
    float *dx_blk = calloc(SD, sizeof(float));
    float *datt = malloc(SD * sizeof(float));
    float *dxn1 = calloc(SD, sizeof(float));
    float *dq = calloc(SD, sizeof(float));
    float *dk = calloc(SD, sizeof(float));
    float *dv_ = calloc(SD, sizeof(float));

    for (int t = 0; t < S; t++)
        for (int v = 0; v < V; v++)
            dlogits[(size_t)t * V + v] =
                (a->probs_final[(size_t)t * V + v] -
                 (v == targets[t] ? 1.0f : 0.0f)) * inv_S;

    /* tied output head */
    for (int t = 0; t < S; t++) {
        const float *dl = dlogits + (size_t)t * V;
        const float *h = a->xhat_final + t * D;
        float *dxh = dxhat + t * D;
        for (int v = 0; v < V; v++) {
            float dlw = dl[v];
            if (dlw == 0.0f) continue;
            float *gr = g->tok_emb->data + (size_t)v * D;
            const float *er = m->tok_emb->data + (size_t)v * D;
            for (int d = 0; d < D; d++) {
                gr[d] += h[d] * dlw;
                dxh[d] += er[d] * dlw;
            }
        }
    }
    for (int t = 0; t < S; t++)
        rms_bwd(dres + t * D, dxhat + t * D, a->x_final + t * D,
                m->final_norm_g, g->final_norm_g, a->rms_f[t], D);

    memcpy(dx_mid, dres, SD * sizeof(float));

    float dgw[F], duw[F], dsw[F];

    for (int l = N_LAYERS - 1; l >= 0; l--) {
        memset(dx_blk, 0, SD * sizeof(float));
        memset(dxn1, 0, SD * sizeof(float));

        /* ---- SwiGLU branch ---- */
        for (int t = 0; t < S; t++) {
            linear_bwd(a->swig[l] + t * F, dx_mid + t * D, g->wd[l], dsw,
                       m->wd[l]);
            float *gp = a->gate_pre[l] + t * F;
            float *up = a->up_pre[l] + t * F;
            for (int f = 0; f < F; f++)
                dgw[f] = dsw[f] * up[f] * silu_grad(gp[f]);
            for (int f = 0; f < F; f++) duw[f] = dsw[f] * silu(gp[f]);
            float acc[D_MODEL];
            memset(acc, 0, sizeof(acc));
            linear_bwd(a->xn2[l] + t * D, dgw, g->wg[l], NULL, m->wg[l]);
            linear_bwd(a->xn2[l] + t * D, duw, g->wu[l], NULL, m->wu[l]);
            /* accumulate wg/wu input-grads into acc manually */
            for (int i = 0; i < D; i++) {
                float s1 = 0.0f, s2 = 0.0f;
                const float *r1 = m->wg[l]->data + (size_t)i * F;
                const float *r2 = m->wu[l]->data + (size_t)i * F;
                for (int f = 0; f < F; f++) { s1 += r1[f]*dgw[f]; s2 += r2[f]*duw[f]; }
                acc[i] = s1 + s2;
            }
            rms_bwd(dx_mid + t * D, acc, a->x_mid[l] + t * D,
                    m->ffn_norm_g[l], g->ffn_norm_g[l], a->rms2[l][t], D);
        }

        /* ---- attention branch ---- */
        memset(datt, 0, SD * sizeof(float));
        for (int t = 0; t < S; t++)
            linear_bwd(a->att[l] + t * D, dx_mid + t * D, g->wo[l],
                       datt + t * D, m->wo[l]);

        memset(dq, 0, SD * sizeof(float));
        memset(dk, 0, SD * sizeof(float));
        memset(dv_, 0, SD * sizeof(float));
        for (int h = 0; h < N_HEADS; h++) {
            int off = h * HEAD_DIM;
            for (int i = 0; i < S; i++) {
                const float *prow =
                    a->probs[l] + ((size_t)h * S + i) * S;
                float dp[SEQ_LEN];
                for (int j = 0; j <= i; j++) {
                    float s_ = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        float da = datt[i * D + off + d];
                        s_ += da * a->v[l][j * D + off + d];
                        dv_[j * D + off + d] += prow[j] * da;
                    }
                    dp[j] = s_;
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
            rope_backward(dq + t * D, t, a);
            rope_backward(dk + t * D, t, a);
        }
        for (int t = 0; t < S; t++) {
            const float *xn = a->xn1[l] + t * D;
            float *dx1 = dxn1 + t * D;
            const float *rq = dq + t * D, *rk = dk + t * D, *rv = dv_ + t * D;
            for (int ii = 0; ii < D; ii++) {
                const float *wq_ = m->wq[l]->data + (size_t)ii * D;
                const float *wk_ = m->wk[l]->data + (size_t)ii * D;
                const float *wv_ = m->wv[l]->data + (size_t)ii * D;
                float s = 0.0f;
                for (int o = 0; o < D; o++)
                    s += wq_[o] * rq[o] + wk_[o] * rk[o] + wv_[o] * rv[o];
                dx1[ii] += s;
            }
            for (int i2 = 0; i2 < D; i2++) {
                float xi = xn[i2];
                float *gwq = g->wq[l]->data + (size_t)i2 * D;
                float *gwk = g->wk[l]->data + (size_t)i2 * D;
                float *gwv = g->wv[l]->data + (size_t)i2 * D;
                for (int o = 0; o < D; o++) {
                    gwq[o] += xi * rq[o];
                    gwk[o] += xi * rk[o];
                    gwv[o] += xi * rv[o];
                }
            }
        }

        for (int t = 0; t < S; t++)
            rms_bwd(dx_blk + t * D, dxn1 + t * D, a->x_layer[l] + t * D,
                    m->attn_norm_g[l], g->attn_norm_g[l], a->rms1[l][t], D);
        vec_add(dx_blk, dx_mid, S * D);
        memcpy(dx_mid, dx_blk, SD * sizeof(float));
    }

    for (int t = 0; t < S; t++)
        vec_add(g->tok_emb->data + tokens[t] * D, dx_blk + t * D, D);

    free(dlogits); free(dxhat); free(dres); free(dx_mid); free(dx_blk);
    free(datt); free(dxn1); free(dq); free(dk); free(dv_);
}

void model_zero_grads(Trainer *tr) { model_zero(&tr->g); }

float model_clip_grads(Trainer *tr, float max_norm) {
    Mat *lst[MAX_PARAMS];
    int n = 0;
    collect(&tr->g, lst, &n);
    double sq = 0.0;
    for (int i = 0; i < n; i++) {
        size_t cnt = (size_t)lst[i]->rows * lst[i]->cols;
        const float *d = lst[i]->data;
        for (size_t e = 0; e < cnt; e++) sq += (double)d[e] * d[e];
    }
    float norm = (float)sqrt(sq);
    if (norm > max_norm && norm > 0.0f) {
        float sc = max_norm / norm;
        for (int i = 0; i < n; i++) {
            size_t cnt = (size_t)lst[i]->rows * lst[i]->cols;
            float *d = lst[i]->data;
            for (size_t e = 0; e < cnt; e++) d[e] *= sc;
        }
        return sc;
    }
    return 1.0f;
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
            w[e] -= lr * ((mm[e] / bc1) / (sqrtf(vv[e] / bc2) + ADAM_EPS) +
                          1e-2f * w[e]);
            if (w[e] > 10.0f) w[e] = 10.0f;
            else if (w[e] < -10.0f) w[e] = -10.0f;
        }
    }
}

int model_save(const Model *mo, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    unsigned magic = CORE_MAGIC, count = count_params();
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
    if (magic != CORE_MAGIC || count != (unsigned)count_params()) {
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
