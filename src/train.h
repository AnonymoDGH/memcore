#ifndef MEMCORE_TRAIN_H
#define MEMCORE_TRAIN_H

#include <stddef.h>

#include "attention.h"
#include "neural_mem.h"
#include "replay.h"

typedef struct {
    Trainer *core;
    NeuralMem *mem;
    ReplayBuf *replay;
    float core_lr;
    float mem_lr_scale;
    float temperature;
    long long step;
    double loss_ema;
} Session;

Session *sess_new(unsigned long long seed);
void sess_free(Session *s);

float sess_train_window(Session *s, const int *window);
float sess_train_bytes(Session *s, const unsigned char *text, size_t len,
                       int max_windows, int verbose_every);

void sess_generate(Session *s, const unsigned char *prompt, size_t plen,
                   int n_gen, unsigned char *out, size_t outcap);

int sess_save(const Session *s, const char *core_path,
              const char *mem_path);
int sess_load(Session *s, const char *core_path, const char *mem_path);

#endif
