#ifndef MEMCORE_TENSOR_H
#define MEMCORE_TENSOR_H

#include <stddef.h>

typedef struct {
    int rows;
    int cols;
    float *data;
} Mat;

Mat *mat_new(int rows, int cols);
void mat_free(Mat *m);
void mat_fill(Mat *m, float v);
void mat_randn(Mat *m, float scale);
float mat_at(const Mat *m, int r, int c);
void mat_set(Mat *m, int r, int c, float v);

void mat_mul(Mat *out, const Mat *a, const Mat *b);
void mat_add_row(Mat *m, int row, const float *v, int n);
void vec_add(float *dst, const float *src, int n);
void vec_scale(float *v, int n, float s);

void softmax_inplace(float *x, int n);
void layer_norm(float *out, const float *in, const float *gamma,
                const float *beta, int n);

extern unsigned long long rng_state;
void rng_seed(unsigned long long seed);
float rng_normal(void);
float rng_uniform(void);

#endif
