#include "tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

unsigned long long rng_state = 88172645463325252ULL;

void rng_seed(unsigned long long seed) {
    rng_state = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

float rng_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)((rng_state >> 11) * (1.0 / 9007199254740992.0));
}

float rng_normal(void) {
    float u1 = rng_uniform();
    float u2 = rng_uniform();
    if (u1 < 1e-12f) u1 = 1e-12f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

Mat *mat_new(int rows, int cols) {
    Mat *m = (Mat *)malloc(sizeof(Mat));
    m->rows = rows;
    m->cols = cols;
    m->data = (float *)calloc((size_t)rows * cols, sizeof(float));
    return m;
}

void mat_free(Mat *m) {
    if (!m) return;
    free(m->data);
    free(m);
}

void mat_fill(Mat *m, float v) {
    size_t n = (size_t)m->rows * m->cols;
    for (size_t i = 0; i < n; i++) m->data[i] = v;
}

void mat_randn(Mat *m, float scale) {
    size_t n = (size_t)m->rows * m->cols;
    for (size_t i = 0; i < n; i++) m->data[i] = rng_normal() * scale;
}

float mat_at(const Mat *m, int r, int c) {
    return m->data[(size_t)r * m->cols + c];
}

void mat_set(Mat *m, int r, int c, float v) {
    m->data[(size_t)r * m->cols + c] = v;
}

void mat_mul(Mat *out, const Mat *a, const Mat *b) {
    for (int i = 0; i < a->rows; i++) {
        float *o = out->data + (size_t)i * out->cols;
        const float *ar = a->data + (size_t)i * a->cols;
        for (int j = 0; j < b->cols; j++) o[j] = 0.0f;
        for (int k = 0; k < a->cols; k++) {
            float a_ik = ar[k];
            const float *br = b->data + (size_t)k * b->cols;
            for (int j = 0; j < b->cols; j++) o[j] += a_ik * br[j];
        }
    }
}

void vec_add(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

void vec_scale(float *v, int n, float s) {
    for (int i = 0; i < n; i++) v[i] *= s;
}

void softmax_inplace(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void layer_norm(float *out, const float *in, const float *gamma,
                const float *beta, int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += in[i];
    mean /= (float)n;

    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
    var /= (float)n;

    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++)
        out[i] = (in[i] - mean) * inv_std * gamma[i] + beta[i];
}
