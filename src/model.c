#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

#define B1 0.9f
#define B2 0.95f
#define EPS 1e-8f
#define MAGIC 0x4D433033u /* "MC03" */

/* ---------- layout ---------- */

static void layout(Model *M) {
    const Config *c = &M->c;
    size_t o = 0, d = (size_t)c->d, ff = (size_t)c->ff;
    M->o_emb = o; o += (size_t)c->vocab * d;
    for (int l = 0; l < c->layers; l++) {
        M->o_g1[l] = o; o += d;
        M->o_qkv[l] = o; o += d * 3 * d;
        M->o_wo[l] = o; o += d * d;
        M->o_g2[l] = o; o += d;
        M->o_wg[l] = o; o += d * ff;
        M->o_wu[l] = o; o += d * ff;
        M->o_wd[l] = o; o += ff * d;
    }
    M->o_gf = o; o += d;
    M->n_params = o;
}

static void rope_init(Model *M) {
    int hd = M->c.d / M->c.heads, half = hd / 2;
    M->rcos = falloc((size_t)M->c.seq * half);
    M->rsin = falloc((size_t)M->c.seq * half);
    for (int t = 0; t < M->c.seq; t++)
        for (int i = 0; i < half; i++) {
            float f = powf(10000.0f, -(float)i / (float)half);
            M->rcos[t * half + i] = cosf((float)t * f);
            M->rsin[t * half + i] = sinf((float)t * f);
        }
}

static Model *model_alloc(Config c) {
    if (c.layers > MC_MAXL || c.seq > MC_MAXT || c.d % c.heads ||
        (c.d / c.heads) % 2)
        return NULL;
    Model *M = calloc(1, sizeof(Model));
    M->c = c;
    layout(M);
    M->p = falloc(M->n_params);
    M->g = falloc(M->n_params);
    M->m = falloc(M->n_params);
    M->v = falloc(M->n_params);
    rope_init(M);
    return M;
}

static void fill_normal(float *p, size_t n, float std) {
    for (size_t i = 0; i < n; i++) p[i] = rng_normal() * std;
}

Model *model_new(Config c, uint64_t seed) {
    Model *M = model_alloc(c);
    if (!M) return NULL;
    rng_seed(seed);
    size_t d = (size_t)c.d, ff = (size_t)c.ff;
    float std = 0.02f;
    float ostd = std / sqrtf(2.0f * (float)(c.layers * c.loops));
    fill_normal(M->p + M->o_emb, (size_t)c.vocab * d, std);
    for (int l = 0; l < c.layers; l++) {
        for (size_t i = 0; i < d; i++) M->p[M->o_g1[l] + i] = 1.0f;
        for (size_t i = 0; i < d; i++) M->p[M->o_g2[l] + i] = 1.0f;
        fill_normal(M->p + M->o_qkv[l], d * 3 * d, std);
        fill_normal(M->p + M->o_wo[l], d * d, ostd);
        fill_normal(M->p + M->o_wg[l], d * ff, std);
        fill_normal(M->p + M->o_wu[l], d * ff, std);
        fill_normal(M->p + M->o_wd[l], ff * d, ostd);
    }
    for (size_t i = 0; i < d; i++) M->p[M->o_gf + i] = 1.0f;
    return M;
}

void model_free(Model *M) {
    if (!M) return;
    free(M->p); free(M->g); free(M->m); free(M->v);
    free(M->rcos); free(M->rsin);
    free(M);
}

int model_save(const Model *M, const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    unsigned magic = MAGIC;
    size_t n = M->n_params;
    int ok = fwrite(&magic, 4, 1, f) == 1 &&
             fwrite(&M->c, sizeof(Config), 1, f) == 1 &&
             fwrite(&M->step, sizeof(M->step), 1, f) == 1 &&
             fwrite(M->meta, sizeof(M->meta), 1, f) == 1 &&
             fwrite(M->p, sizeof(float), n, f) == n &&
             fwrite(M->m, sizeof(float), n, f) == n &&
             fwrite(M->v, sizeof(float), n, f) == n;
    fclose(f);
    if (!ok) return -1;
    return rename(tmp, path);
}

Model *model_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned magic = 0;
    Config c;
    if (fread(&magic, 4, 1, f) != 1 || magic != MAGIC ||
        fread(&c, sizeof(c), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    Model *M = model_alloc(c);
    size_t n = M ? M->n_params : 0;
    int ok = M && fread(&M->step, sizeof(M->step), 1, f) == 1 &&
             fread(M->meta, sizeof(M->meta), 1, f) == 1 &&
             fread(M->p, sizeof(float), n, f) == n &&
             fread(M->m, sizeof(float), n, f) == n &&
             fread(M->v, sizeof(float), n, f) == n;
    fclose(f);
    if (!ok) { model_free(M); return NULL; }
    return M;
}

void model_print(const Model *M) {
    const Config *c = &M->c;
    printf("MemCore v3: d=%d heads=%d blocks=%d x loops<=%d (depth<=%d) "
           "ff=%d seq=%d vocab=%d\n",
           c->d, c->heads, c->layers, c->loops, c->layers * c->loops, c->ff,
           c->seq, c->vocab);
    printf("  params: %zu (%.2f MB fp32), trained steps: %lld\n", M->n_params,
           (double)M->n_params * 4.0 / 1048576.0, M->step);
}

/* ---------- workspace ---------- */

Work *work_new(const Model *M, int B, int T) {
    const Config *c = &M->c;
    if (T > c->seq) return NULL;
    Work *w = calloc(1, sizeof(Work));
    size_t N = (size_t)B * T, d = (size_t)c->d, ff = (size_t)c->ff;
    size_t A = (size_t)c->layers * c->loops;
    w->B = B; w->T = T; w->N = (int)N; w->A = (int)A;
    w->tok = calloc(N, sizeof(int));
    w->tgt = calloc(N, sizeof(int));
    w->x = falloc((A + 1) * N * d);
    w->xn1 = falloc(A * N * d);
    w->qkv = falloc(A * N * 3 * d);
    w->att = falloc(A * N * d);
    w->probs = falloc(A * (size_t)B * c->heads * T * T);
    w->xmid = falloc(A * N * d);
    w->xn2 = falloc(A * N * d);
    w->gp = falloc(A * N * ff);
    w->up = falloc(A * N * ff);
    w->hh = falloc(A * N * ff);
    w->r1 = falloc(A * N);
    w->r2 = falloc(A * N);
    w->xnf = falloc(N * d);
    w->rf = falloc(N);
    w->logits = falloc(N * (size_t)c->vocab);
    w->dx = falloc(N * d);
    w->dxn = falloc(N * d);
    w->dqkv = falloc(N * 3 * d);
    w->datt = falloc(N * d);
    w->dhh = falloc(N * ff);
    w->dgp = falloc(N * ff);
    w->dup = falloc(N * ff);
    return w;
}

void work_free(Work *w) {
    if (!w) return;
    float *bufs[] = {w->x, w->xn1, w->qkv, w->att, w->probs, w->xmid,
                     w->xn2, w->gp, w->up, w->hh, w->r1, w->r2, w->xnf,
                     w->rf, w->logits, w->dx, w->dxn, w->dqkv, w->datt,
                     w->dhh, w->dgp, w->dup};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) free(bufs[i]);
    free(w->tok); free(w->tgt);
    free(w);
}

/* ---------- primitives ---------- */

static void rms_fwd(float *y, const float *x, const float *g, int N, int d,
                    float *rstd) {
#pragma omp parallel for if (N >= 64)
    for (int n = 0; n < N; n++) {
        const float *xr = x + (size_t)n * d;
        float *yr = y + (size_t)n * d;
        float ss = 0;
        for (int i = 0; i < d; i++) ss += xr[i] * xr[i];
        float r = 1.0f / sqrtf(ss / (float)d + 1e-5f);
        rstd[n] = r;
        for (int i = 0; i < d; i++) yr[i] = xr[i] * r * g[i];
    }
}

/* dx += d(rmsnorm)/dx ; dg += sum_n dy*xhat */
static void rms_bwd(float *dx, const float *dy, const float *x, const float *g,
                    float *dg, const float *rstd, int N, int d) {
#pragma omp parallel for if (N >= 64)
    for (int n = 0; n < N; n++) {
        const float *xr = x + (size_t)n * d, *dyr = dy + (size_t)n * d;
        float *dxr = dx + (size_t)n * d;
        float r = rstd[n], s = 0;
        for (int i = 0; i < d; i++) s += g[i] * dyr[i] * xr[i];
        float k = r * r * r * s / (float)d;
        for (int i = 0; i < d; i++) dxr[i] += r * g[i] * dyr[i] - xr[i] * k;
    }
    for (int n = 0; n < N; n++) {
        const float *xr = x + (size_t)n * d, *dyr = dy + (size_t)n * d;
        float r = rstd[n];
        for (int i = 0; i < d; i++) dg[i] += dyr[i] * xr[i] * r;
    }
}

static void rope(float *v, int t, const Model *M, int inverse) {
    int H = M->c.heads, hd = M->c.d / H, half = hd / 2;
    const float *cs = M->rcos + (size_t)t * half, *sn = M->rsin + (size_t)t * half;
    for (int h = 0; h < H; h++) {
        float *p = v + h * hd;
        for (int i = 0; i < half; i++) {
            float c = cs[i], s = inverse ? -sn[i] : sn[i];
            float a = p[i], b = p[i + half];
            p[i] = a * c - b * s;
            p[i + half] = a * s + b * c;
        }
    }
}

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ---------- forward ---------- */

float model_forward(Model *M, Work *w, int loops) {
    const Config *c = &M->c;
    int N = w->N, T = w->T, B = w->B, d = c->d, H = c->heads, hd = d / H;
    int ff = c->ff, V = c->vocab, d3 = 3 * d;
    if (loops < 1) loops = 1;
    if (loops > c->loops) loops = c->loops;
    int A = c->layers * loops;
    w->A_run = A;
    float scale = 1.0f / sqrtf((float)hd);
    const float *P = M->p;

    for (int n = 0; n < N; n++)
        memcpy(w->x + (size_t)n * d, P + M->o_emb + (size_t)w->tok[n] * d,
               sizeof(float) * d);

    for (int a = 0; a < A; a++) {
        int l = a % c->layers;
        size_t Nd = (size_t)N * d, Nf = (size_t)N * ff;
        float *xi = w->x + a * Nd, *xo = w->x + (a + 1) * Nd;
        float *xn1 = w->xn1 + a * Nd, *qkv = w->qkv + a * Nd * 3;
        float *att = w->att + a * Nd, *xm = w->xmid + a * Nd;
        float *xn2 = w->xn2 + a * Nd;
        float *pr = w->probs + (size_t)a * B * H * T * T;
        float *gp = w->gp + a * Nf, *up = w->up + a * Nf, *hh = w->hh + a * Nf;

        rms_fwd(xn1, xi, P + M->o_g1[l], N, d, w->r1 + (size_t)a * N);
        mm_fwd(qkv, xn1, P + M->o_qkv[l], N, d, d3, 0);
#pragma omp parallel for if (N >= 64)
        for (int n = 0; n < N; n++) {
            rope(qkv + (size_t)n * d3, n % T, M, 0);
            rope(qkv + (size_t)n * d3 + d, n % T, M, 0);
        }
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++) {
                for (int t = 0; t < T; t++) {
                    const float *q = qkv + (size_t)(b * T + t) * d3 + h * hd;
                    float *prow = pr + ((size_t)(b * H + h) * T + t) * T;
                    float mx = -1e30f;
                    for (int j = 0; j <= t; j++) {
                        const float *k =
                            qkv + (size_t)(b * T + j) * d3 + d + h * hd;
                        float s = 0;
                        for (int i = 0; i < hd; i++) s += q[i] * k[i];
                        s *= scale;
                        prow[j] = s;
                        if (s > mx) mx = s;
                    }
                    float sum = 0;
                    for (int j = 0; j <= t; j++) {
                        prow[j] = expf(prow[j] - mx);
                        sum += prow[j];
                    }
                    float inv = 1.0f / sum;
                    for (int j = 0; j <= t; j++) prow[j] *= inv;
                    for (int j = t + 1; j < T; j++) prow[j] = 0;
                    float *o = att + (size_t)(b * T + t) * d + h * hd;
                    for (int i = 0; i < hd; i++) o[i] = 0;
                    for (int j = 0; j <= t; j++) {
                        const float *v =
                            qkv + (size_t)(b * T + j) * d3 + 2 * d + h * hd;
                        float p = prow[j];
                        for (int i = 0; i < hd; i++) o[i] += p * v[i];
                    }
                }
            }
        mm_fwd(xm, att, P + M->o_wo[l], N, d, d, 0);
        for (size_t i = 0; i < Nd; i++) xm[i] += xi[i];
        rms_fwd(xn2, xm, P + M->o_g2[l], N, d, w->r2 + (size_t)a * N);
        mm_fwd(gp, xn2, P + M->o_wg[l], N, d, ff, 0);
        mm_fwd(up, xn2, P + M->o_wu[l], N, d, ff, 0);
        for (size_t i = 0; i < Nf; i++) hh[i] = gp[i] * sigmoidf(gp[i]) * up[i];
        mm_fwd(xo, hh, P + M->o_wd[l], N, ff, d, 0);
        for (size_t i = 0; i < Nd; i++) xo[i] += xm[i];
    }

    float *xA = w->x + (size_t)A * N * d;
    rms_fwd(w->xnf, xA, P + M->o_gf, N, d, w->rf);
    mm_bt(w->logits, w->xnf, P + M->o_emb, N, V, d, 0);

    double loss = 0;
    int cnt = 0;
#pragma omp parallel for reduction(+ : loss, cnt) if (N >= 64)
    for (int n = 0; n < N; n++) {
        float *row = w->logits + (size_t)n * V;
        float mx = row[0];
        for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
        float sum = 0;
        for (int v = 0; v < V; v++) { row[v] = expf(row[v] - mx); sum += row[v]; }
        float inv = 1.0f / sum;
        for (int v = 0; v < V; v++) row[v] *= inv;
        if (w->tgt[n] >= 0) {
            loss -= log((double)fmaxf(row[w->tgt[n]], 1e-30f));
            cnt++;
        }
    }
    w->nvalid = cnt;
    return cnt ? (float)(loss / cnt) : 0.0f;
}

/* ---------- backward ---------- */

void model_backward(Model *M, Work *w) {
    const Config *c = &M->c;
    int N = w->N, T = w->T, B = w->B, d = c->d, H = c->heads, hd = d / H;
    int ff = c->ff, V = c->vocab, d3 = 3 * d, A = w->A_run;
    if (w->nvalid == 0) return;
    float scale = 1.0f / sqrtf((float)hd);
    const float *P = M->p;
    float *G = M->g;
    float inv = 1.0f / (float)w->nvalid;
    size_t Nd = (size_t)N * d, Nf = (size_t)N * ff;

    float *dl = w->logits; /* probs -> dlogits in place */
#pragma omp parallel for if (N >= 64)
    for (int n = 0; n < N; n++) {
        float *row = dl + (size_t)n * V;
        if (w->tgt[n] < 0) {
            memset(row, 0, sizeof(float) * V);
        } else {
            for (int v = 0; v < V; v++) row[v] *= inv;
            row[w->tgt[n]] -= inv;
        }
    }
    mm_fwd(w->dxn, dl, P + M->o_emb, N, V, d, 0);
    mm_wgrad(G + M->o_emb, dl, w->xnf, N, V, d);
    memset(w->dx, 0, sizeof(float) * Nd);
    rms_bwd(w->dx, w->dxn, w->x + (size_t)A * Nd, P + M->o_gf, G + M->o_gf,
            w->rf, N, d);

    for (int a = A - 1; a >= 0; a--) {
        int l = a % c->layers;
        float *xi = w->x + a * Nd;
        float *xn1 = w->xn1 + a * Nd, *qkv = w->qkv + a * Nd * 3;
        float *att = w->att + a * Nd, *xm = w->xmid + a * Nd;
        float *xn2 = w->xn2 + a * Nd;
        float *pr = w->probs + (size_t)a * B * H * T * T;
        float *gp = w->gp + a * Nf, *up = w->up + a * Nf, *hh = w->hh + a * Nf;

        /* FFN */
        mm_bt(w->dhh, w->dx, P + M->o_wd[l], N, ff, d, 0);
        mm_wgrad(G + M->o_wd[l], hh, w->dx, N, ff, d);
        for (size_t i = 0; i < Nf; i++) {
            float g = gp[i], s = sigmoidf(g), dh = w->dhh[i];
            w->dup[i] = dh * g * s;
            w->dgp[i] = dh * up[i] * s * (1.0f + g * (1.0f - s));
        }
        mm_bt(w->dxn, w->dgp, P + M->o_wg[l], N, d, ff, 0);
        mm_bt(w->dxn, w->dup, P + M->o_wu[l], N, d, ff, 1);
        mm_wgrad(G + M->o_wg[l], xn2, w->dgp, N, d, ff);
        mm_wgrad(G + M->o_wu[l], xn2, w->dup, N, d, ff);
        rms_bwd(w->dx, w->dxn, xm, P + M->o_g2[l], G + M->o_g2[l],
                w->r2 + (size_t)a * N, N, d);

        /* attention */
        mm_bt(w->datt, w->dx, P + M->o_wo[l], N, d, d, 0);
        mm_wgrad(G + M->o_wo[l], att, w->dx, N, d, d);
        memset(w->dqkv, 0, sizeof(float) * Nd * 3);
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int b = 0; b < B; b++)
            for (int h = 0; h < H; h++) {
                float dp[MC_MAXT];
                for (int t = 0; t < T; t++) {
                    const float *dout = w->datt + (size_t)(b * T + t) * d + h * hd;
                    const float *prow = pr + ((size_t)(b * H + h) * T + t) * T;
                    const float *q = qkv + (size_t)(b * T + t) * d3 + h * hd;
                    float *dq = w->dqkv + (size_t)(b * T + t) * d3 + h * hd;
                    float sum = 0;
                    for (int j = 0; j <= t; j++) {
                        const float *v =
                            qkv + (size_t)(b * T + j) * d3 + 2 * d + h * hd;
                        float *dv =
                            w->dqkv + (size_t)(b * T + j) * d3 + 2 * d + h * hd;
                        float s = 0;
                        for (int i = 0; i < hd; i++) {
                            s += dout[i] * v[i];
                            dv[i] += prow[j] * dout[i];
                        }
                        dp[j] = s;
                        sum += prow[j] * s;
                    }
                    for (int j = 0; j <= t; j++) {
                        float ds = prow[j] * (dp[j] - sum) * scale;
                        const float *k =
                            qkv + (size_t)(b * T + j) * d3 + d + h * hd;
                        float *dk = w->dqkv + (size_t)(b * T + j) * d3 + d + h * hd;
                        for (int i = 0; i < hd; i++) {
                            dq[i] += ds * k[i];
                            dk[i] += ds * q[i];
                        }
                    }
                }
            }
#pragma omp parallel for if (N >= 64)
        for (int n = 0; n < N; n++) {
            rope(w->dqkv + (size_t)n * d3, n % T, M, 1);
            rope(w->dqkv + (size_t)n * d3 + d, n % T, M, 1);
        }
        mm_bt(w->dxn, w->dqkv, P + M->o_qkv[l], N, d, d3, 0);
        mm_wgrad(G + M->o_qkv[l], xn1, w->dqkv, N, d, d3);
        rms_bwd(w->dx, w->dxn, xi, P + M->o_g1[l], G + M->o_g1[l],
                w->r1 + (size_t)a * N, N, d);
    }

    for (int n = 0; n < N; n++) {
        float *ge = G + M->o_emb + (size_t)w->tok[n] * d;
        const float *dxr = w->dx + (size_t)n * d;
        for (int i = 0; i < d; i++) ge[i] += dxr[i];
    }
}

/* ---------- optimizer ---------- */

void model_zero_grad(Model *M) { memset(M->g, 0, sizeof(float) * M->n_params); }

float model_clip(Model *M, float max_norm) {
    double ss = 0;
    size_t n = M->n_params;
#pragma omp parallel for reduction(+ : ss)
    for (size_t i = 0; i < n; i++) ss += (double)M->g[i] * M->g[i];
    float norm = (float)sqrt(ss);
    if (norm > max_norm) {
        float s = max_norm / (norm + 1e-6f);
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) M->g[i] *= s;
    }
    return norm;
}

static void adam_range(Model *M, size_t o, size_t n, float lr, float wd,
                       float bc1, float bc2) {
    float *p = M->p + o, *g = M->g + o, *m = M->m + o, *v = M->v + o;
#pragma omp parallel for if (n >= 4096)
    for (size_t i = 0; i < n; i++) {
        m[i] = B1 * m[i] + (1 - B1) * g[i];
        v[i] = B2 * v[i] + (1 - B2) * g[i] * g[i];
        float mh = m[i] / bc1, vh = v[i] / bc2;
        p[i] -= lr * (mh / (sqrtf(vh) + EPS) + wd * p[i]);
    }
}

void model_adamw(Model *M, float lr, float wd) {
    M->step++;
    float bc1 = 1.0f - powf(B1, (float)M->step);
    float bc2 = 1.0f - powf(B2, (float)M->step);
    const Config *c = &M->c;
    size_t d = (size_t)c->d, ff = (size_t)c->ff;
    adam_range(M, M->o_emb, (size_t)c->vocab * d, lr, wd, bc1, bc2);
    for (int l = 0; l < c->layers; l++) {
        adam_range(M, M->o_g1[l], d, lr, 0, bc1, bc2);
        adam_range(M, M->o_qkv[l], 3 * d * d, lr, wd, bc1, bc2);
        adam_range(M, M->o_wo[l], d * d, lr, wd, bc1, bc2);
        adam_range(M, M->o_g2[l], d, lr, 0, bc1, bc2);
        adam_range(M, M->o_wg[l], 3 * d * ff, lr, wd, bc1, bc2); /* wg,wu,wd */
    }
    adam_range(M, M->o_gf, d, lr, 0, bc1, bc2);
}

/* ---------- incremental decoding ---------- */

KV *kv_new(const Model *M) {
    const Config *c = &M->c;
    KV *kv = calloc(1, sizeof(KV));
    kv->A = c->layers * c->loops;
    kv->seq = c->seq;
    kv->d = c->d;
    kv->ff = c->ff;
    size_t sz = (size_t)kv->A * c->seq * c->d;
    kv->k = falloc(sz);
    kv->v = falloc(sz);
    kv->x = falloc(c->d);
    kv->xn = falloc(c->d);
    kv->qkv = falloc(3 * (size_t)c->d);
    kv->att = falloc(c->d);
    kv->tmp = falloc(c->d);
    kv->gp = falloc(c->ff);
    kv->up = falloc(c->ff);
    kv->sc = falloc(c->seq);
    return kv;
}

void kv_free(KV *kv) {
    if (!kv) return;
    free(kv->k); free(kv->v); free(kv->x); free(kv->xn); free(kv->qkv);
    free(kv->att); free(kv->tmp); free(kv->gp); free(kv->up); free(kv->sc);
    free(kv);
}

void model_step(const Model *M, KV *kv, int tok, int pos, int loops,
                float *logits, float *hidden) {
    const Config *c = &M->c;
    int d = c->d, H = c->heads, hd = d / H, ff = c->ff, d3 = 3 * d;
    if (loops < 1) loops = 1;
    if (loops > c->loops) loops = c->loops;
    int A = c->layers * loops;
    float scale = 1.0f / sqrtf((float)hd);
    const float *P = M->p;
    float r;

    memcpy(kv->x, P + M->o_emb + (size_t)tok * d, sizeof(float) * d);
    for (int a = 0; a < A; a++) {
        int l = a % c->layers;
        float *K = kv->k + (size_t)a * kv->seq * d;
        float *Vv = kv->v + (size_t)a * kv->seq * d;
        rms_fwd(kv->xn, kv->x, P + M->o_g1[l], 1, d, &r);
        mm_fwd(kv->qkv, kv->xn, P + M->o_qkv[l], 1, d, d3, 0);
        rope(kv->qkv, pos, M, 0);
        rope(kv->qkv + d, pos, M, 0);
        memcpy(K + (size_t)pos * d, kv->qkv + d, sizeof(float) * d);
        memcpy(Vv + (size_t)pos * d, kv->qkv + 2 * d, sizeof(float) * d);
        for (int h = 0; h < H; h++) {
            const float *q = kv->qkv + h * hd;
            float mx = -1e30f;
            for (int j = 0; j <= pos; j++) {
                const float *k = K + (size_t)j * d + h * hd;
                float s = 0;
                for (int i = 0; i < hd; i++) s += q[i] * k[i];
                kv->sc[j] = s * scale;
                if (kv->sc[j] > mx) mx = kv->sc[j];
            }
            float sum = 0;
            for (int j = 0; j <= pos; j++) {
                kv->sc[j] = expf(kv->sc[j] - mx);
                sum += kv->sc[j];
            }
            float *o = kv->att + h * hd;
            for (int i = 0; i < hd; i++) o[i] = 0;
            for (int j = 0; j <= pos; j++) {
                const float *v = Vv + (size_t)j * d + h * hd;
                float p = kv->sc[j] / sum;
                for (int i = 0; i < hd; i++) o[i] += p * v[i];
            }
        }
        mm_fwd(kv->tmp, kv->att, P + M->o_wo[l], 1, d, d, 0);
        for (int i = 0; i < d; i++) kv->x[i] += kv->tmp[i];
        rms_fwd(kv->xn, kv->x, P + M->o_g2[l], 1, d, &r);
        mm_fwd(kv->gp, kv->xn, P + M->o_wg[l], 1, d, ff, 0);
        mm_fwd(kv->up, kv->xn, P + M->o_wu[l], 1, d, ff, 0);
        for (int i = 0; i < ff; i++)
            kv->gp[i] = kv->gp[i] * sigmoidf(kv->gp[i]) * kv->up[i];
        mm_fwd(kv->tmp, kv->gp, P + M->o_wd[l], 1, ff, d, 0);
        for (int i = 0; i < d; i++) kv->x[i] += kv->tmp[i];
    }
    rms_fwd(kv->xn, kv->x, P + M->o_gf, 1, d, &r);
    if (hidden) memcpy(hidden, kv->xn, sizeof(float) * d);
    if (logits) mm_bt(logits, kv->xn, P + M->o_emb, 1, c->vocab, d, 0);
}
