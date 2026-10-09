#ifndef MC_MEMORY_H
#define MC_MEMORY_H

#include <stddef.h>
#include "model.h"

/* Episodic kNN memory: (hidden state -> next byte) pairs written instantly,
   no gradient step. Mixed into the model's distribution at decode time,
   gated by how close the nearest stored context is. */
typedef struct {
    int d, n, cap;
    float *keys;
    unsigned char *vals;
} KMem;

KMem *kmem_new(int d);
void kmem_free(KMem *km);
int kmem_save(const KMem *km, const char *path);
KMem *kmem_load(const char *path);
void kmem_add(KMem *km, const float *key, int val);
/* runs the model over text and stores every (context, next byte) pair */
void kmem_learn(KMem *km, const Model *M, KV *kv, const unsigned char *text,
                size_t len);
/* probs <- (1-g) probs + g p_knn, g = lambda * closeness; returns g */
float kmem_mix(const KMem *km, const float *h, float *probs, int V,
               float lambda);

#endif
