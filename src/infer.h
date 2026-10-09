#ifndef MC_INFER_H
#define MC_INFER_H

#include <stddef.h>
#include "memory.h"
#include "model.h"

typedef struct {
    char ans[64];
    float conf;  /* min token probability along the answer */
    float agree; /* fraction of votes for the winning answer */
    int loops;   /* depth actually used */
} Solution;

/* decode one answer after "\n<prompt>" until '\n' */
void infer_answer(const Model *M, KV *kv, const char *prompt, int loops,
                  float temp, char *ans, size_t cap, float *conf);

/* adaptive compute: shallow pass first, think deeper when unsure;
   votes > 1 adds sampled answers and takes the majority */
void infer_solve(const Model *M, KV *kv, const char *prompt, int votes,
                 Solution *out);

/* free text generation, optionally mixing episodic memory */
size_t infer_generate(const Model *M, KV *kv, const KMem *mem,
                      const unsigned char *prompt, size_t plen, int n,
                      float temp, unsigned char *out, size_t cap);

#endif
