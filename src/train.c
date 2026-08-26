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
    s->use_replay = 1;
    s->use_mem = 1;
    return s;
}

void sess_free(Session *s) {
    if (!s) return;
    trainer_free(s->core);
    nmem_free(s->mem);
    rb_free(s->replay);
    free(s);
}

static void shift_targets(const int *window, int *tgt) {
    for (int t = 0; t < SEQ_LEN - 1; t++) tgt[t] = window[t + 1];
    tgt[SEQ_LEN - 1] = window[SEQ_LEN - 1];
}

float sess_train_batch(Session *s, int wins[][SEQ_LEN + 1], int nb) {
    Trainer *tr = s->core;
    Activations *a = &tr->act;

    model_zero_grads(tr);
    double loss = 0.0;
    for (int k = 0; k < nb; k++) {
        shift_targets(wins[k], dummy_targets);
        loss += model_forward(tr, wins[k], dummy_targets, NULL);
        model_backward(tr, wins[k], dummy_targets);

        for (int t = 0; t < SEQ_LEN; t++) {
            int tgt = (t + 1 < SEQ_LEN) ? wins[k][t + 1] : wins[k][t];
            nmem_surprise_update(s->mem, a->x_final + t * D_MODEL, tgt,
                                 s->mem_lr_scale);
        }
        rb_push(s->replay, wins[k]);
    }

    model_clip_grads(tr, 1.0f);
    float warm = fminf(1.0f, (float)(s->step + 1) / 300.0f);
    model_adam_step(tr, s->core_lr * warm);

    float avg = (float)(loss / nb);
    s->loss_ema =
        s->loss_ema == 0.0 ? avg : 0.95f * s->loss_ema + 0.05f * avg;
    s->step += nb;
    return avg;
}

float sess_train_window(Session *s, const int *window) {
    int one[1][SEQ_LEN + 1];
    memcpy(one[0], window, sizeof(int) * (SEQ_LEN + 1));
    return sess_train_batch(s, one, 1);
}

float sess_train_bytes(Session *s, const unsigned char *text, size_t len,
                       int max_windows, int verbose_every) {
    int *ids = malloc(sizeof(int) * (len + 16));
    if (!ids) return 0.0f;
    long ntok = tok_encode(text, len, ids, len + 15);
    if (ntok < SEQ_LEN + 1) { free(ids); return 0.0f; }

    long pos = 0;
    int win_count = 0;
    double acc = 0.0;
    int batches = 0;

    while (pos + SEQ_LEN + 1 <= ntok && win_count < max_windows) {
        int wins[8][SEQ_LEN + 1];
        int nb = 0;
        while (nb < 6 && pos + SEQ_LEN + 1 <= ntok &&
               win_count + nb < max_windows) {
            for (int t = 0; t < SEQ_LEN + 1; t++)
                wins[nb][t] = ids[pos + t];
            nb++;
            pos += SEQ_LEN / 2;
        }
        if (s->use_replay && s->replay->count > 32) {
            for (int r = 0; r < 2 && nb < 8; r++) {
                rb_sample(s->replay, wins[nb], NULL);
                nb++;
            }
        }
        acc += sess_train_batch(s, wins, nb);
        batches++;
        win_count += nb;
        if (verbose_every > 0 && batches % verbose_every == 0)
            printf("  [%6d win] loss=%.4f ema=%.4f\n", win_count,
                   acc / batches, s->loss_ema);
    }
    free(ids);
    return batches ? (float)(acc / batches) : 0.0f;
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
                   int n_gen, unsigned char *out, size_t outcap,
                   int use_mem) {
    int stream[8192];
    int len = 0;
    if (plen > 2048) plen = 2048;
    if (len < 8192) stream[len++] = '\n';
    if (s->replay->count > 0) {
        int last =
            (s->replay->head - 1 + s->replay->capacity) % s->replay->capacity;
        const int *w = s->replay->slots + (size_t)last * SEQ_LEN;
        for (int i = 0; i < SEQ_LEN && len < 4096; i++) stream[len++] = w[i];
        if (len < 8192) stream[len++] = '\n';
    }
    for (size_t k = 0; k < plen && len < 8192; k++)
        stream[len++] = (unsigned char)prompt[k];
    size_t written = 0;
    int gids[4096];
    int ngid = 0;
    for (int g = 0; g < n_gen && written < outcap; g++) {
        int lo = len > SEQ_LEN ? len - SEQ_LEN : 0;
        int valid = len - lo;
        int un = 0;
        int use[SEQ_LEN];
        for (int i = lo; i < len; i++) use[un++] = stream[i];
        while (un < SEQ_LEN) use[un++] = ' ';
        int R = valid - 1;

        int dtargets[SEQ_LEN];
        for (int t = 0; t < SEQ_LEN - 1; t++) dtargets[t] = use[t + 1];
        dtargets[SEQ_LEN - 1] = use[SEQ_LEN - 1];

        model_forward(s->core, use, dtargets, NULL);

        const float *p_core =
            s->core->act.probs_final + (size_t)R * VOCAB_SIZE;
        if (getenv("MEMCORE_DEBUG") && g == 0) {
            float lchk = 0.0f;
            for (int t = 0; t < SEQ_LEN; t++) {
                const float *pp =
                    s->core->act.probs_final + (size_t)t * VOCAB_SIZE;
                lchk -= logf(fmaxf(pp[dtargets[t]], 1e-10f));
            }
            fprintf(stderr, "[dbg] R=%d len=%d ce_last=%.3f\n", R, len,
                    lchk / SEQ_LEN);
        }
        float combined[VOCAB_SIZE];
        float zmax = -1e30f;
        if (use_mem) {
            float xlast[D_MODEL];
            memcpy(xlast, s->core->act.x_final + (size_t)R * D_MODEL,
                   sizeof(float) * D_MODEL);
            float mem_out[VOCAB_SIZE];
            nmem_forward(s->mem, xlast, mem_out);
            float mmax = -1e30f;
            for (int v = 0; v < VOCAB_SIZE; v++)
                if (mem_out[v] > mmax) mmax = mem_out[v];
            double msum = 0.0;
            for (int v = 0; v < VOCAB_SIZE; v++) {
                mem_out[v] = expf(mem_out[v] - mmax);
                msum += mem_out[v];
            }
            for (int v = 0; v < VOCAB_SIZE; v++) {
                float lp_core = logf(fmaxf(p_core[v], 1e-12f));
                float lp_mem =
                    logf(fmaxf((float)(mem_out[v] / msum), 1e-12f));
                combined[v] = 0.85f * lp_core + 0.15f * lp_mem / 3.0f;
                if (combined[v] > zmax) zmax = combined[v];
            }
        } else {
            for (int v = 0; v < VOCAB_SIZE; v++) {
                combined[v] = logf(fmaxf(p_core[v], 1e-12f));
                if (combined[v] > zmax) zmax = combined[v];
            }
        }
        float sum = 0.0f;
        for (int v = 0; v < VOCAB_SIZE; v++) {
            combined[v] = expf(combined[v] - zmax);
            sum += combined[v];
        }
        for (int v = 0; v < VOCAB_SIZE; v++) combined[v] /= sum;

        if (getenv("MEMCORE_DEBUG") && g == 0) {
            int bi[5] = {0};
            float bc[5];
            for (int k = 0; k < 5; k++) {
                bi[k] = 0;
                for (int v = 1; v < VOCAB_SIZE; v++)
                    if (combined[v] > combined[bi[k]]) bi[k] = v;
                bc[k] = combined[bi[k]];
                for (int kk = 0; kk <= k; kk++) combined[bi[kk]] = -1;
            }
            fprintf(stderr, "[dbg] R=%d top:", R);
            for (int k = 0; k < 5; k++) fprintf(stderr, " %d(%.3f)", bi[k], bc[k]);
            fprintf(stderr, "\n");
            /* restore */
            for (int k = 0; k < 5; k++) combined[bi[k]] = bc[k];
        }
        int next;
        sample_from_probs(combined, s->temperature, &next);
        if (len < 8192) stream[len++] = next;
        if (ngid < 4096) gids[ngid++] = next;

        if (!tok_ready()) out[written++] = (unsigned char)next;
    }
    if (tok_ready() && ngid > 0)
        written += tok_decode(gids, ngid, out + written,
                              outcap > written ? outcap - written : 0);
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
