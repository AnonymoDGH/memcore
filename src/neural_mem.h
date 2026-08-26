#ifndef MEMCORE_NEURAL_MEM_H
#define MEMCORE_NEURAL_MEM_H

#include "config.h"

typedef struct {
    float *W1;
    float *b1;
    float *W2;
    float *b2;
    float lr;
    float decay_lambda;
    float last_surprise;
    long long updates;
} NeuralMem;

NeuralMem *nmem_new(unsigned long long seed);
void nmem_free(NeuralMem *nm);

void nmem_forward(const NeuralMem *nm, const float *x, float *out);
float nmem_forward_batch(const NeuralMem *nm, const float *x /*[S*D]*/,
                         float *out /*[S*V]*/);

float nmem_surprise_update(NeuralMem *nm, const float *x, int target_token,
                           float lr_scale);

int nmem_save(const NeuralMem *nm, const char *path);
NeuralMem *nmem_load(const char *path);

#endif
