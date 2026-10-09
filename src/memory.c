#include "memory.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

#define KNN_K 8
#define KMEM_MAGIC 0x4B4D454Du

KMem *kmem_new(int d) {
    KMem *km = calloc(1, sizeof(KMem));
    km->d = d;
    return km;
}

void kmem_free(KMem *km) {
    if (!km) return;
    free(km->keys);
    free(km->vals);
    free(km);
}

void kmem_add(KMem *km, const float *key, int val) {
    if (km->n == km->cap) {
        int nc = km->cap ? km->cap * 2 : 1024;
        km->keys = realloc(km->keys, sizeof(float) * (size_t)nc * km->d);
        km->vals = realloc(km->vals, (size_t)nc);
        km->cap = nc;
    }
    memcpy(km->keys + (size_t)km->n * km->d, key, sizeof(float) * km->d);
    km->vals[km->n++] = (unsigned char)val;
}

int kmem_save(const KMem *km, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    unsigned magic = KMEM_MAGIC;
    size_t n = (size_t)km->n;
    int ok = fwrite(&magic, 4, 1, f) == 1 && fwrite(&km->d, 4, 1, f) == 1 &&
             fwrite(&km->n, 4, 1, f) == 1 &&
             fwrite(km->keys, sizeof(float) * km->d, n, f) == n &&
             fwrite(km->vals, 1, n, f) == n;
    fclose(f);
    return ok ? 0 : -1;
}

KMem *kmem_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned magic = 0;
    int d = 0, n = 0;
    if (fread(&magic, 4, 1, f) != 1 || magic != KMEM_MAGIC ||
        fread(&d, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    KMem *km = kmem_new(d);
    km->cap = n > 0 ? n : 1;
    km->keys = malloc(sizeof(float) * (size_t)km->cap * d);
    km->vals = malloc((size_t)km->cap);
    km->n = n;
    int ok = fread(km->keys, sizeof(float) * d, (size_t)n, f) == (size_t)n &&
             fread(km->vals, 1, (size_t)n, f) == (size_t)n;
    fclose(f);
    if (!ok) { kmem_free(km); return NULL; }
    return km;
}

void kmem_learn(KMem *km, const Model *M, KV *kv, const unsigned char *text,
                size_t len) {
    int seq = M->c.seq, half = seq / 2;
    float *h = falloc((size_t)M->c.d);
    /* overlapping windows so every stored context has >= half a window */
    for (size_t s = 0; s + 1 < len; s += (size_t)half) {
        size_t keep_from = s == 0 ? 0 : s + (size_t)half;
        for (int p = 0; p < seq && s + (size_t)p + 1 < len; p++) {
            size_t i = s + (size_t)p;
            model_step(M, kv, text[i], p, M->c.loops, NULL, h);
            if (i >= keep_from) kmem_add(km, h, text[i + 1]);
        }
        if (s + (size_t)seq >= len) break;
    }
    free(h);
}

float kmem_mix(const KMem *km, const float *h, float *probs, int V,
               float lambda) {
    if (!km || km->n == 0) return 0.0f;
    float bd[KNN_K] = {0};
    int bi[KNN_K], nb = 0;
    for (int i = 0; i < km->n; i++) {
        const float *k = km->keys + (size_t)i * km->d;
        float s = 0;
        for (int j = 0; j < km->d; j++) { float e = h[j] - k[j]; s += e * e; }
        if (nb < KNN_K) { bd[nb] = s; bi[nb++] = i; continue; }
        int w = 0;
        for (int j = 1; j < KNN_K; j++) if (bd[j] > bd[w]) w = j;
        if (s < bd[w]) { bd[w] = s; bi[w] = i; }
    }
    float dmin = bd[0];
    for (int j = 1; j < nb; j++) if (bd[j] < dmin) dmin = bd[j];
    float tau = 0.05f * (float)km->d;
    float pk[256] = {0}, z = 0;
    for (int j = 0; j < nb; j++) {
        float wgt = expf(-(bd[j] - dmin) / tau);
        pk[km->vals[bi[j]]] += wgt;
        z += wgt;
    }
    float g = lambda * expf(-dmin / tau);
    for (int v = 0; v < V && v < 256; v++)
        probs[v] = (1.0f - g) * probs[v] + g * pk[v] / z;
    return g;
}
