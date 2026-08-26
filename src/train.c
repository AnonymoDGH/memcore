#include "train.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
#include "tokenizer.h"

static int dummy_targets[SEQ_LEN];

Session *sess_new(unsigned long long seed) {
    rng_seed(seed);
    Session *s = calloc(1, sizeof(Session));
    s->core = trainer_new();
    s->mem = nmem_new(seed ^ 0xB5297A4DULL);
    s->replay = rb_new(512);
    s->core_lr = 3e-4f;
    s->mem_lr_scale = 1.0f;
    s->temperature = 0.8f;
    return s;
}

void sess_free(Session *s) {
    if (!s) return;
    trainer_free(s->core);
    nmem_free(s->mem);
    rb_free(s->replay);
    free(s);
}

float sess_train_window(Session *s, const int *window) {
    Trainer *tr = s->core;
    Activations *a = &tr->act;

    for (int t = 0; t < SEQ_LEN - 1; t++)
        dummy_targets[t] = window[t + 1];
    dummy_targets[SEQ_LEN - 1] = window[SEQ_LEN - 1];

    model_zero_grads(tr);
    float loss =
        model_forward(tr, window, dummy_targets, NULL);
    model_backward(tr, window, dummy_targets);
    model_adam_step(tr, s->core_lr);

    float surprise_sum = 0.0f;
    for (int t = 0; t < SEQ_LEN; t++) {
        int tgt = (t + 1 < SEQ_LEN) ? window[t + 1] : window[t];
        surprise_sum += nmem_surprise_update(
            s->mem, a->x_final + t * D_MODEL, tgt, s->mem_lr_scale);
    }

    rb_push(s->replay, window);
    if (s->step % 2 == 1 && s->replay->count > 8) {
        int rwin[SEQ_LEN];
        for (int k = 0; k < 2; k++) {
            rb_sample(s->replay, rwin, NULL);
            for (int t = 0; t < SEQ_LEN - 1; t++)
                dummy_targets[t] = rwin[t + 1];
            dummy_targets[SEQ_LEN - 1] = rwin[SEQ_LEN - 1];
            model_zero_grads(tr);
            float rloss = model_forward(tr, rwin, dummy_targets, NULL);
            model_backward(tr, rwin, dummy_targets);
            model_adam_step(tr, s->core_lr * 0.5f);
            loss = 0.7f * loss + 0.3f * rloss;
            for (int t = 0; t < SEQ_LEN; t++) {
                int tgt = (t + 1 < SEQ_LEN) ? rwin[t + 1] : rwin[t];
                nmem_surprise_update(s->mem, a->x_final + t * D_MODEL, tgt,
                                     s->mem_lr_scale * 0.5f);
            }
        }
    }

    s->loss_ema = s->loss_ema == 0.0 ? loss : 0.95f * s->loss_ema + 0.05f * loss;
    s->step++;
    return loss;
}

float sess_train_bytes(Session *s, const unsigned char *text, size_t len,
                       int max_windows, int verbose_every) {
    size_t total = len;
    if (total < (size_t)(SEQ_LEN + 1)) return 0.0f;

    size_t pos = 0;
    int win_idx = 0;
    double acc = 0.0;
    int count = 0;
    while (pos + SEQ_LEN + 1 <= total && win_idx < max_windows) {
        int window[SEQ_LEN + 1];
        tok_encode(text + pos, SEQ_LEN + 1, window, SEQ_LEN + 1);
        acc += sess_train_window(s, window);
        count++;
        win_idx++;
        pos += SEQ_LEN / 2;
        if (verbose_every > 0 && count % verbose_every == 0)
            printf("  [%6d] loss=%.4f ema=%.4f mem_surprise=%.3f\n", count,
                   acc / count, s->loss_ema, s->mem->last_surprise);
    }
    return count ? (float)(acc / count) : 0.0f;
}

static void sample_from_probs(const float *probs, float temperature,
                              int *out_tok) {
    if (temperature <= 0.01f) {
        int best = 0;
        for (int v = 1; v < VOCAB_SIZE; v++)
            if (probs[v] > probs[best]) best = v;
        *out_tok = best;
        return;
    }
    float scaled[VOCAB_SIZE];
    float sum = 0.0f;
    float inv_t = 1.0f / temperature;
    for (int v = 0; v < VOCAB_SIZE; v++) {
        scaled[v] = powf(probs[v], inv_t);
        sum += scaled[v];
    }
    float r = rng_uniform() * sum;
    float cdf = 0.0f;
    for (int v = 0; v < VOCAB_SIZE; v++) {
        cdf += scaled[v];
        if (r <= cdf) {
            *out_tok = v;
            return;
        }
    }
    *out_tok = VOCAB_SIZE - 1;
}

void sess_generate(Session *s, const unsigned char *prompt, size_t plen,
                   int n_gen, unsigned char *out, size_t outcap) {
    int ctx[SEQ_LEN];
    memset(ctx, 0, sizeof(ctx));

    int seed_tokens[SEQ_LEN];
    int got = tok_encode(prompt, plen, seed_tokens, SEQ_LEN);

    int start = 0;
    for (int i = 0; i < got && start < SEQ_LEN; i++)
        ctx[start++] = seed_tokens[i];

    size_t written = 0;
    for (int g = 0; g < n_gen; g++) {
        int lo = start > SEQ_LEN ? start - SEQ_LEN : 0;
        int use[SEQ_LEN];
        int un = 0;
        for (int i = lo; i < start; i++) use[un++] = ctx[i % SEQ_LEN];
        while (un < SEQ_LEN) use[un++] = 0;

        int dtargets[SEQ_LEN];
        for (int t = 0; t < SEQ_LEN - 1; t++) dtargets[t] = use[t + 1];
        dtargets[SEQ_LEN - 1] = use[SEQ_LEN - 1];

        model_forward(s->core, use, dtargets, NULL);

        const float *p_core = s->core->act.probs_final +
                              (size_t)(SEQ_LEN - 1) * VOCAB_SIZE;
        float xlast[D_MODEL];
        memcpy(xlast,
               s->core->act.x_final + (size_t)(SEQ_LEN - 1) * D_MODEL,
               sizeof(float) * D_MODEL);
        float mem_out[VOCAB_SIZE];
        nmem_forward(s->mem, xlast, mem_out);

        float combined[VOCAB_SIZE];
        float zmax = -1e30f;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            combined[v] = logf(fmaxf(p_core[v], 1e-12f)) + mem_out[v];
            if (combined[v] > zmax) zmax = combined[v];
        }
        float sum = 0.0f;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            combined[v] = expf(combined[v] - zmax);
            sum += combined[v];
        }
        for (int v = 0; v < VOCAB_SIZE; v++) combined[v] /= sum;

        int next;
        sample_from_probs(combined, s->temperature, &next);
        ctx[start % SEQ_LEN] = next;
        start++;

        if (written < outcap) out[written++] = (unsigned char)next;
        if (written >= outcap) break;
    }
    if (written < outcap) out[written] = '\0';
    else if (outcap > 0) out[outcap - 1] = '\0';
}

int sess_save(const Session *s, const char *core_path, const char *mem_path) {
    if (model_save(&s->core->m, core_path) != 0) return -1;
    if (nmem_save(s->mem, mem_path) != 0) return -1;
    return 0;
}

int sess_load(Session *s, const char *core_path, const char *mem_path) {
    if (model_load(&s->core->m, core_path) != 0) return -1;
    NeuralMem *nm = nmem_load(mem_path);
    if (!nm) return -1;
    nmem_free(s->mem);
    s->mem = nm;
    return 0;
}
