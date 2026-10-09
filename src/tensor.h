#ifndef MC_TENSOR_H
#define MC_TENSOR_H

#include <stddef.h>
#include <stdint.h>

void rng_seed(uint64_t s);
uint64_t rng_u64(void);
float rng_uniform(void); /* [0,1) */
float rng_normal(void);
int rng_int(int n); /* [0,n) */

/* zeroed, 64-byte aligned */
float *falloc(size_t n);

/* y[N,out] (= or +=) x[N,in] @ W[in,out] */
void mm_fwd(float *y, const float *x, const float *W, int N, int in, int out,
            int acc);
/* y[N,in] (= or +=) x[N,out] @ W[in,out]^T */
void mm_bt(float *y, const float *x, const float *W, int N, int in, int out,
           int acc);
/* dW[in,out] += x[N,in]^T @ dy[N,out] */
void mm_wgrad(float *dW, const float *x, const float *dy, int N, int in,
              int out);

#endif
