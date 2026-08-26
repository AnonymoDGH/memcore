#include "neural_mem.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

#define NMEM_MAGIC 0x4D454D43u

NeuralMem *nmem_new(unsigned long long seed) {
    NeuralMem *nm = calloc(1, sizeof(NeuralMem));
    size_t nW1 = (size_t)D_MODEL * MEM_HIDDEN;
    size_t nW2 = (size_t)MEM_HIDDEN * VOCAB_SIZE;
    nm->W1 = malloc(nW1 * sizeof(float));
    nm->b1 = calloc(MEM_HIDDEN, sizeof(float));
    nm->W2 = malloc(nW2 * sizeof(float));
    nm->b2 = calloc(VOCAB_SIZE, sizeof(float));

    rng_seed(seed);
    float s1 = sqrtf(2.0f / (float)D_MODEL);
    for (size_t i = 0; i < nW1; i++) nm->W1[i] = rng_normal() * s1;
    float s2 = 1.0f / sqrtf((float)MEM_HIDDEN);
    for (size_t i = 0; i < nW2; i++) nm->W2[i] = rng_normal() * s2;

    nm->lr = 0.02f;
    nm->decay_lambda = 2.0f;
    return nm;
}

void nmem_free(NeuralMem *nm) {
    if (!nm) return;
    free(nm->W1); free(nm->b1); free(nm->W2); free(nm->b2);
    free(nm);
}

static void mem_forward_raw(const NeuralMem *nm, const float *x, float *h,
                            float *out) {
    for (int j = 0; j < MEM_HIDDEN; j++) {
        const float *wr = nm->W1 + (size_t)j * D_MODEL;
        float acc = nm->b1[j];
        for (int d = 0; d < D_MODEL; d++) acc += x[d] * wr[d];
        h[j] = tanhf(acc);
    }
    for (int v = 0; v < VOCAB_SIZE; v++) out[v] = nm->b2[v];
    for (int j = 0; j < MEM_HIDDEN; j++) {
        float hj = h[j];
        const float *wr = nm->W2 + (size_t)j * VOCAB_SIZE;
        for (int v = 0; v < VOCAB_SIZE; v++) out[v] += hj * wr[v];
    }
}

void nmem_forward(const NeuralMem *nm, const float *x, float *out) {
    float h[MEM_HIDDEN];
    mem_forward_raw(nm, x, h, out);
}

float nmem_forward_batch(const NeuralMem *nm, const float *x, float *out) {
    float h[MEM_HIDDEN];
    int S = SEQ_LEN;
    for (int t = 0; t < S; t++)
        mem_forward_raw(nm, x + t * D_MODEL, h, out + (size_t)t * VOCAB_SIZE);
    return 0.0f;
}

float nmem_surprise_update(NeuralMem *nm, const float *x, int target_token,
                           float lr_scale) {
    float h[MEM_HIDDEN];
    float out[VOCAB_SIZE];
    mem_forward_raw(nm, x, h, out);

    float maxv = out[0];
    for (int v = 1; v < VOCAB_SIZE; v++)
        if (out[v] > maxv) maxv = out[v];
    float sum = 0.0f;
    float p[VOCAB_SIZE];
    for (int v = 0; v < VOCAB_SIZE; v++) {
        p[v] = expf(out[v] - maxv);
        sum += p[v];
    }
    for (int v = 0; v < VOCAB_SIZE; v++) p[v] /= sum;

    float loss = -logf(fmaxf(p[target_token], 1e-10f));
    float surprise = fminf(loss / 5.0f, 1.0f);

    float lr = nm->lr * lr_scale;
    float decay = (1.0f - nm->decay_lambda * surprise * lr) *
                  (1.0f - 2e-4f);
    if (decay < 0.0f) decay = 0.0f;
    nm->lr = 0.02f;

    float gvec[VOCAB_SIZE];
    for (int v = 0; v < VOCAB_SIZE; v++)
        gvec[v] = (p[v] - (v == target_token ? 1.0f : 0.0f)) * lr;

    float dh[MEM_HIDDEN];
    for (int j = 0; j < MEM_HIDDEN; j++) {
        const float *wr = nm->W2 + (size_t)j * VOCAB_SIZE;
        float acc = 0.0f;
        for (int v = 0; v < VOCAB_SIZE; v++) acc += gvec[v] * wr[v];
        dh[j] = acc;
    }

    for (int v = 0; v < VOCAB_SIZE; v++)
        nm->b2[v] = nm->b2[v] * decay - gvec[v];

    for (int j = 0; j < MEM_HIDDEN; j++) {
        float hj = h[j];
        float *wr = nm->W2 + (size_t)j * VOCAB_SIZE;
        for (int v = 0; v < VOCAB_SIZE; v++) wr[v] = wr[v] * decay - hj * gvec[v];
    }

    for (int j = 0; j < MEM_HIDDEN; j++) {
        float t = h[j];
        float dt = (1.0f - t * t) * dh[j];
        nm->b1[j] = nm->b1[j] * decay - dt;
        for (int d = 0; d < D_MODEL; d++)
            nm->W1[(size_t)j * D_MODEL + d] =
                nm->W1[(size_t)j * D_MODEL + d] * decay - x[d] * dt;
    }

    nm->last_surprise = surprise;
    nm->updates++;
    return surprise;
}

int nmem_save(const NeuralMem *nm, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    unsigned magic = NMEM_MAGIC;
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&nm->lr, sizeof(float), 1, f);
    fwrite(&nm->decay_lambda, sizeof(float), 1, f);
    fwrite(nm->W1, sizeof(float), (size_t)D_MODEL * MEM_HIDDEN, f);
    fwrite(nm->b1, sizeof(float), MEM_HIDDEN, f);
    fwrite(nm->W2, sizeof(float), (size_t)MEM_HIDDEN * VOCAB_SIZE, f);
    fwrite(nm->b2, sizeof(float), VOCAB_SIZE, f);
    fclose(f);
    return 0;
}

NeuralMem *nmem_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned magic = 0;
    fread(&magic, sizeof(magic), 1, f);
    if (magic != NMEM_MAGIC) { fclose(f); return NULL; }
    NeuralMem *nm = calloc(1, sizeof(NeuralMem));
    fread(&nm->lr, sizeof(float), 1, f);
    fread(&nm->decay_lambda, sizeof(float), 1, f);
    nm->W1 = malloc((size_t)D_MODEL * MEM_HIDDEN * sizeof(float));
    nm->b1 = malloc(MEM_HIDDEN * sizeof(float));
    nm->W2 = malloc((size_t)MEM_HIDDEN * VOCAB_SIZE * sizeof(float));
    nm->b2 = malloc(VOCAB_SIZE * sizeof(float));
    fread(nm->W1, sizeof(float), (size_t)D_MODEL * MEM_HIDDEN, f);
    fread(nm->b1, sizeof(float), MEM_HIDDEN, f);
    fread(nm->W2, sizeof(float), (size_t)MEM_HIDDEN * VOCAB_SIZE, f);
    fread(nm->b2, sizeof(float), VOCAB_SIZE, f);
    fclose(f);
    return nm;
}
