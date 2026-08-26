#ifndef MEMCORE_CONFIG_H
#define MEMCORE_CONFIG_H

#define VOCAB_SIZE 256
#define D_MODEL 64
#define N_HEADS 4
#define N_LAYERS 2
#define SEQ_LEN 64
#define FF_HIDDEN (4 * D_MODEL)
#define MEM_HIDDEN 128
#define HEAD_DIM (D_MODEL / N_HEADS)

#endif
