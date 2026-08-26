#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "attention.h"
#include "tensor.h"
#include "tokenizer.h"
#include "train.h"

int main(void) {
    rng_seed(42);
    Session *s = sess_new(42);
    if (tok_ensure("data/train.txt", "data/bpe.bin") != 0) return 1;
    printf("[tok %s]\n", tok_name());

    FILE *f = fopen("data/train.txt", "rb");
    static unsigned char buf[65536];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);

    /* train a few hundred windows IN PROCESS */
    int *ids = malloc(sizeof(int) * (n + 16));
    long nt = tok_encode(buf, n, ids, n + 15);
    printf("ntok=%ld\n", nt);
    long pos = 0;
    float lsum = 0;
    for (int w = 0; w < 400 && pos + SEQ_LEN + 1 <= nt; w++) {
        int win[SEQ_LEN + 1];
        memcpy(win, ids + pos, sizeof(int) * (SEQ_LEN + 1));
        lsum += sess_train_window(s, win);
        pos += SEQ_LEN / 2;
    }
    printf("avg train loss=%.3f\n", lsum / 400.0f);

    /* same-process generation */
    unsigned char out[300];
    sess_generate(s, (const unsigned char *)"el sol sale por la", 18, 60,
                  out, sizeof(out) - 1, 1);
    out[sizeof(out) - 1] = 0;
    printf("gen: [%s]\n", (char *)out);

    /* teacher-forced: what does the model predict after real context? */
    long q = pos - 40 > 0 ? pos - 40 : 0;
    if (q + SEQ_LEN <= nt) {
        int win[SEQ_LEN];
        memcpy(win, ids + q, SEQ_LEN * sizeof(int));
        int tg[SEQ_LEN];
        for (int t = 0; t < SEQ_LEN - 1; t++) tg[t] = win[t + 1];
        tg[SEQ_LEN - 1] = win[SEQ_LEN - 1];
        model_forward(s->core, win, tg, NULL);
        const float *pp = s->core->act.probs_final + 30 * VOCAB_SIZE;
        int b = 0;
        for (int v = 1; v < VOCAB_SIZE; v++)
            if (pp[v] > pp[b]) b = v;
        unsigned char dec[8];
        tok_decode(&win[31], 1, dec, 8);
        tok_decode(&b, 1, dec + 4, 4);
        printf("t-forced@30: in='%c' -> pred_id=%d '%.*s' p=%.3f\n", dec[0],
               b, 4, dec + 4, pp[b]);
    }
    sess_free(s);
    free(ids);
    return 0;
}
