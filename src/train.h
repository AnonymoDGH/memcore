#ifndef MC_TRAIN_H
#define MC_TRAIN_H

#include <stddef.h>
#include "model.h"

typedef struct {
    int steps, B, T, eval_every, log_every;
    float lr, wd;
    const unsigned char *text; /* optional corpus mixed into batches */
    size_t text_len;
    float text_mix;
    const char *ckpt;
} TrainOpts;

void train_run(Model *M, const TrainOpts *o);
/* accuracy table per task and level, beyond the training ceiling too */
float eval_report(Model *M, int n, int votes, int extra_levels);

#endif
