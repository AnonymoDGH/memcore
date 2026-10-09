#include "tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t g_state = 0x9E3779B97F4A7C15ULL;

void rng_seed(uint64_t s) { g_state = s ? s : 1; }

uint64_t rng_u64(void) {
    uint64_t z = (g_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

float rng_uniform(void) { return (float)(rng_u64() >> 40) * (1.0f / 16777216.0f); }

int rng_int(int n) { return n <= 1 ? 0 : (int)(rng_u64() % (uint64_t)n); }

float rng_normal(void) {
    float u1 = rng_uniform(), u2 = rng_uniform();
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}

float *falloc(size_t n) {
    size_t bytes = (n ? n : 1) * sizeof(float);
    bytes = (bytes + 63) & ~(size_t)63;
    float *p = aligned_alloc(64, bytes);
    if (p) memset(p, 0, bytes);
    return p;
}

void mm_fwd(float *restrict y, const float *restrict x,
            const float *restrict W, int N, int in, int out, int acc) {
#pragma omp parallel for schedule(static) if (N >= 32)
    for (int n0 = 0; n0 < N; n0 += 4) {
        int nb = N - n0 < 4 ? N - n0 : 4;
        float *y0 = y + (size_t)n0 * out;
        if (!acc) memset(y0, 0, sizeof(float) * (size_t)nb * out);
        if (nb == 4) {
            float *y1 = y0 + out, *y2 = y1 + out, *y3 = y2 + out;
            const float *x0 = x + (size_t)n0 * in, *x1 = x0 + in,
                        *x2 = x1 + in, *x3 = x2 + in;
            for (int i = 0; i < in; i++) {
                const float *w = W + (size_t)i * out;
                float a0 = x0[i], a1 = x1[i], a2 = x2[i], a3 = x3[i];
#pragma omp simd
                for (int o = 0; o < out; o++) {
                    float wv = w[o];
                    y0[o] += a0 * wv;
                    y1[o] += a1 * wv;
                    y2[o] += a2 * wv;
                    y3[o] += a3 * wv;
                }
            }
        } else {
            for (int r = 0; r < nb; r++) {
                float *yr = y0 + (size_t)r * out;
                const float *xr = x + (size_t)(n0 + r) * in;
                for (int i = 0; i < in; i++) {
                    const float *w = W + (size_t)i * out;
                    float a = xr[i];
#pragma omp simd
                    for (int o = 0; o < out; o++) yr[o] += a * w[o];
                }
            }
        }
    }
}

void mm_bt(float *restrict y, const float *restrict x,
           const float *restrict W, int N, int in, int out, int acc) {
#pragma omp parallel for schedule(static) if (N >= 32)
    for (int n0 = 0; n0 < N; n0 += 4) {
        int nb = N - n0 < 4 ? N - n0 : 4;
        if (nb == 4) {
            const float *x0 = x + (size_t)n0 * out, *x1 = x0 + out,
                        *x2 = x1 + out, *x3 = x2 + out;
            float *y0 = y + (size_t)n0 * in, *y1 = y0 + in, *y2 = y1 + in,
                  *y3 = y2 + in;
            for (int i = 0; i < in; i++) {
                const float *w = W + (size_t)i * out;
                float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma omp simd reduction(+ : s0, s1, s2, s3)
                for (int o = 0; o < out; o++) {
                    float wv = w[o];
                    s0 += x0[o] * wv;
                    s1 += x1[o] * wv;
                    s2 += x2[o] * wv;
                    s3 += x3[o] * wv;
                }
                if (acc) {
                    y0[i] += s0; y1[i] += s1; y2[i] += s2; y3[i] += s3;
                } else {
                    y0[i] = s0; y1[i] = s1; y2[i] = s2; y3[i] = s3;
                }
            }
        } else {
            for (int r = 0; r < nb; r++) {
                const float *xr = x + (size_t)(n0 + r) * out;
                float *yr = y + (size_t)(n0 + r) * in;
                for (int i = 0; i < in; i++) {
                    const float *w = W + (size_t)i * out;
                    float s = 0;
#pragma omp simd reduction(+ : s)
                    for (int o = 0; o < out; o++) s += xr[o] * w[o];
                    yr[i] = acc ? yr[i] + s : s;
                }
            }
        }
    }
}

#define WG_TILE 256

void mm_wgrad(float *restrict dW, const float *restrict x,
              const float *restrict dy, int N, int in, int out) {
    int nblk = (in + 3) / 4;
#pragma omp parallel if (N * in >= 4096)
    for (int t0 = 0; t0 < N; t0 += WG_TILE) {
        int t1 = t0 + WG_TILE < N ? t0 + WG_TILE : N;
        /* static schedule: each thread keeps the same rows of dW in every
           tile, so nowait is race-free */
#pragma omp for schedule(static) nowait
        for (int b = 0; b < nblk; b++) {
            int i0 = b * 4;
            int ib = in - i0 < 4 ? in - i0 : 4;
            if (ib == 4) {
                float *d0 = dW + (size_t)i0 * out, *d1 = d0 + out,
                      *d2 = d1 + out, *d3 = d2 + out;
                for (int n = t0; n < t1; n++) {
                    const float *xr = x + (size_t)n * in + i0;
                    const float *g = dy + (size_t)n * out;
                    float a0 = xr[0], a1 = xr[1], a2 = xr[2], a3 = xr[3];
#pragma omp simd
                    for (int o = 0; o < out; o++) {
                        float gv = g[o];
                        d0[o] += a0 * gv;
                        d1[o] += a1 * gv;
                        d2[o] += a2 * gv;
                        d3[o] += a3 * gv;
                    }
                }
            } else {
                for (int r = 0; r < ib; r++) {
                    float *dr = dW + (size_t)(i0 + r) * out;
                    for (int n = t0; n < t1; n++) {
                        float a = x[(size_t)n * in + i0 + r];
                        const float *g = dy + (size_t)n * out;
#pragma omp simd
                        for (int o = 0; o < out; o++) dr[o] += a * g[o];
                    }
                }
            }
        }
    }
}
