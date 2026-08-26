#include "replay.h"

#include <stdlib.h>
#include <string.h>
#include "tensor.h"

ReplayBuf *rb_new(int capacity) {
    ReplayBuf *rb = calloc(1, sizeof(ReplayBuf));
    rb->capacity = capacity;
    rb->slots = calloc((size_t)capacity * SEQ_LEN, sizeof(int));
    return rb;
}

void rb_free(ReplayBuf *rb) {
    if (!rb) return;
    free(rb->slots);
    free(rb);
}

void rb_push(ReplayBuf *rb, const int *window) {
    memcpy(rb->slots + (size_t)rb->head * SEQ_LEN, window,
           SEQ_LEN * sizeof(int));
    rb->head = (rb->head + 1) % rb->capacity;
    if (rb->count < rb->capacity) rb->count++;
}

void rb_sample(const ReplayBuf *rb, int *out_window, float *randval) {
    if (rb->count == 0) return;
    int idx = (int)(rng_uniform() * (float)rb->count) % rb->count;
    memcpy(out_window, rb->slots + (size_t)idx * SEQ_LEN,
           SEQ_LEN * sizeof(int));
}
