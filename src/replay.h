#ifndef MEMCORE_REPLAY_H
#define MEMCORE_REPLAY_H

#include "config.h"

typedef struct {
    int *slots;
    int capacity;
    int count;
    int head;
} ReplayBuf;

ReplayBuf *rb_new(int capacity);
void rb_free(ReplayBuf *rb);
void rb_push(ReplayBuf *rb, const int *window);
void rb_sample(const ReplayBuf *rb, int *out_window, float *randval);

#endif
