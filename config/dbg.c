#include <stdio.h>
#include <string.h>

#include "attention.h"
#include "neural_mem.h"
#include "tensor.h"
#include "train.h"

int main(void) {
    Session *s = sess_new(42);
    if (sess_load(s, "model.core", "model.mem") != 0) {
        printf("no checkpoint\n");
        return 1;
    }
    printf("checkpoint loaded ok\n");

    const char *corpus =
        "xxxe sol sale por la manana y el cielo se vuelve azul\n";
    int window[SEQ_LEN];
    int targets[SEQ_LEN];
    int n = 0;
    for (const char *p = corpus; *p && n < SEQ_LEN; p++)
        window[n++] = (unsigned char)*p;
    while (n < SEQ_LEN) window[n++] = ' ';
    for (int t = 0; t < SEQ_LEN - 1; t++) targets[t] = window[t + 1];
    targets[SEQ_LEN - 1] = window[SEQ_LEN - 1];

    float loss = model_forward(s->core, window, targets, NULL);
    printf("loss on known prefix: %.4f\n", loss);

    for (int t = 45; t < SEQ_LEN; t++) {
        const float *p = s->core->act.probs_final + (size_t)t * VOCAB_SIZE;
        int best = 0;
        for (int v = 1; v < VOCAB_SIZE; v++)
            if (p[v] > p[best]) best = v;
        printf("pos %2d in='%c' target='%c' argmax='%c' p=%.3f\n", t,
               window[t], targets[t], best, p[best]);
    }

    unsigned char out[200];
    memset(out, 0, sizeof(out));
    printf("--- sess_generate direct call ---\n");
    s->temperature = 0.5f;
    sess_generate(s, (const unsigned char *)"el sol sale por la", 18, 60,
                  out, 100, 0);
    printf("gen output: [%s]\n", (char *)out);

    sess_free(s);
    return 0;
}
