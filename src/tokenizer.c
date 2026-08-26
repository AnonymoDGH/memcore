#include "tokenizer.h"

#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_MERGES (VOCAB_SIZE - 256)
#define HBITS 17
#define HSIZE (1 << HBITS)
#define TOK_MAGIC 0x42504532u
#define MAX_TOK_LEN 40

typedef struct {
    int a, b;
} Pair;

static Pair g_merges[MAX_MERGES];
static int g_n_merges = 0;
static unsigned char g_str[VOCAB_SIZE][MAX_TOK_LEN + 1];
static unsigned char g_len[VOCAB_SIZE];
static int g_ready = 0;

static unsigned long long h_key[HSIZE];
static int h_cnt[HSIZE];

static unsigned long long pair_key(int a, int b) {
    return ((unsigned long long)a << 20) | (unsigned)b;
}

static unsigned h_slot(unsigned long long k) {
    return (unsigned)((k * 0x9E3779B97F4A7C15ULL) >> (64 - HBITS)) & (HSIZE - 1);
}

static void init_base(void) {
    for (int i = 0; i < 256; i++) {
        g_str[i][0] = (unsigned char)i;
        g_len[i] = 1;
    }
}

static int tok_is_space(unsigned char c) {
    return c == ' ' || c == '\n' || c == '\t' || c == '\r';
}

static void h_add(unsigned long long k, int v, int *best_cnt,
                  unsigned long long *best_k) {
    unsigned s = h_slot(k);
    for (;;) {
        if (h_cnt[s] == 0) {
            h_key[s] = k;
            h_cnt[s] = v;
            if (v > *best_cnt) { *best_cnt = v; *best_k = k; }
            return;
        }
        if (h_key[s] == k) {
            h_cnt[s] += v;
            if (h_cnt[s] > *best_cnt) { *best_cnt = h_cnt[s]; *best_k = k; }
            return;
        }
        s = (s + 1) & (HSIZE - 1);
    }
}

static int merge_slice(int *x, int n, int a, int b, int ab) {
    int j = 0;
    for (int i = 0; i < n;) {
        if (i + 1 < n && x[i] == a && x[i + 1] == b) {
            x[j++] = ab;
            i += 2;
        } else {
            x[j++] = x[i++];
        }
    }
    return j;
}

static void commit_merge(int m, int a, int b) {
    int id = 256 + m;
    int la = g_len[a], lb = g_len[b];
    g_merges[m].a = a;
    g_merges[m].b = b;
    if (la + lb <= MAX_TOK_LEN) {
        memcpy(g_str[id], g_str[a], la);
        memcpy(g_str[id] + la, g_str[b], lb);
        g_len[id] = (unsigned char)(la + lb);
    } else {
        g_len[id] = 255;
    }
}

typedef struct {
    int *ids;
    int *starts;
    int *ends;
    int nseg;
} SegStream;

static void train_merges_on_stream(SegStream *ss, const char *save_path,
                                   int verbose_every) {
    memset(h_cnt, 0, sizeof(h_cnt));
    for (int m = 0; m < MAX_MERGES; m++) {
        memset(h_cnt, 0, sizeof(h_cnt));
        int best_cnt = 1;
        unsigned long long best_k = 0;
        for (int sgi = 0; sgi < ss->nseg; sgi++) {
            int s = ss->starts[sgi], e = ss->ends[sgi];
            for (int i = s; i < e - 1; i++)
                h_add(pair_key(ss->ids[i], ss->ids[i + 1]), 1, &best_cnt,
                      &best_k);
        }
        if (best_cnt < 2 || best_k == 0) break;
        int a = (int)(best_k >> 20), b = (int)(best_k & 0xFFFFF);
        commit_merge(m, a, b);
        g_n_merges = m + 1;
        int id = 256 + m;
        for (int sgi = 0; sgi < ss->nseg; sgi++) {
            int s = ss->starts[sgi], e = ss->ends[sgi];
            if (e - s < 2) continue;
            int nn = merge_slice(ss->ids + s, e - s, a, b, id);
            ss->ends[sgi] = s + nn;
        }
        if (verbose_every > 0 && (m + 1) % verbose_every == 0)
            printf("[bpe] merge %d: '%.*s'+'%.*s' x%d\n", m + 1,
                   g_len[a], g_str[a], g_len[b], g_str[b], best_cnt);
    }
    FILE *f = fopen(save_path, "wb");
    if (f) {
        unsigned magic = TOK_MAGIC;
        fwrite(&magic, sizeof(magic), 1, f);
        fwrite(&g_n_merges, sizeof(int), 1, f);
        fwrite(g_merges, sizeof(Pair), (size_t)g_n_merges, f);
        fclose(f);
    }
}

static int next_chunk(const unsigned char *text, size_t len, size_t i,
                      size_t *chunk_end) {
    while (i < len && (text[i] == ' ' || text[i] == '\t')) i++;
    if (i >= len) return 0;
    if (text[i] == '\n' || text[i] == '\r') {
        *chunk_end = i + 1;
        return 1;
    }
    size_t j = i;
    while (j < len && !tok_is_space(text[j])) j++;
    *chunk_end = j;
    return j > i ? 1 : 0;
}

static void apply_all_merges(int *x, int n) {
    for (int m = 0; m < g_n_merges; m++) {
        int a = g_merges[m].a, b = g_merges[m].b;
        int hit = 0;
        for (int i = 0; i < n - 1; i++)
            if (x[i] == a && x[i + 1] == b) { hit = 1; break; }
        if (!hit) continue;
        n = merge_slice(x, n, a, b, 256 + m);
    }
}

int tok_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    unsigned magic = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1 ||
        magic != TOK_MAGIC ||
        fread(&g_n_merges, sizeof(int), 1, f) != 1 ||
        g_n_merges <= 0 || g_n_merges > MAX_MERGES) {
        fclose(f);
        return -1;
    }
    if (fread(g_merges, sizeof(Pair), (size_t)g_n_merges, f) !=
        (size_t)g_n_merges) {
        fclose(f);
        return -1;
    }
    fclose(f);
    init_base();
    for (int m = 0; m < g_n_merges; m++)
        commit_merge(m, g_merges[m].a, g_merges[m].b);
    g_ready = 1;
    return 0;
}

int tok_ensure(const char *corpus_path, const char *model_path) {
    if (tok_load(model_path) == 0) return 0;

    FILE *f = fopen(corpus_path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return -1; }
    if (sz > 120000) sz = 120000;
    unsigned char *raw = malloc((size_t)sz);
    size_t rd = fread(raw, 1, (size_t)sz, f);
    fclose(f);

    SegStream ss;
    ss.ids = malloc(sizeof(int) * rd);
    ss.starts = malloc(sizeof(int) * (rd / 2 + 16));
    ss.ends = malloc(sizeof(int) * (rd / 2 + 16));
    ss.nseg = 0;

    size_t p = 0;
    long n = 0;
    int cap = (int)(rd / 2 + 16);
    while (p < rd && ss.nseg < cap) {
        size_t ce;
        if (!next_chunk(raw, rd, p, &ce)) break;
        int st = (int)n;
        for (size_t k = p; k < ce; k++) ss.ids[n++] = raw[k];
        ss.starts[ss.nseg] = st;
        ss.ends[ss.nseg] = (int)n;
        ss.nseg++;
        p = ce;
    }

    init_base();
    g_n_merges = 0;
    train_merges_on_stream(&ss, model_path, 128);
    printf("[bpe] trained %d merges -> %s\n", g_n_merges, model_path);

    free(ss.ids); free(ss.starts); free(ss.ends); free(raw);
    g_ready = 1;
    return 0;
}

int tok_ready(void) { return g_ready; }
int tok_vocab(void) { return VOCAB_SIZE; }

int tok_encode(const unsigned char *text, size_t len, int *out,
               size_t max_tokens) {
    if (!g_ready) {
        size_t c = len < max_tokens ? len : max_tokens;
        for (size_t i = 0; i < c; i++) out[i] = text[i];
        return (int)c;
    }
    int w = 0;
    size_t p = 0;
    int scratch[MAX_TOK_LEN + 8];
    while (p < len && w < (int)max_tokens) {
        size_t ce;
        if (!next_chunk(text, len, p, &ce)) break;
        int n = 0;
        for (size_t k = p; k < ce && n < MAX_TOK_LEN + 4; k++)
            scratch[n++] = text[k];
        apply_all_merges(scratch, n);
        for (int i = 0; i < n && w < (int)max_tokens; i++)
            out[w++] = scratch[i];
        p = ce;
    }
    return w;
}

int tok_decode(const int *tokens, size_t n, unsigned char *out,
               size_t max_bytes) {
    size_t w = 0;
    for (size_t i = 0; i < n && w < max_bytes; i++) {
        int t = tokens[i];
        if (t < 0 || t >= VOCAB_SIZE) { out[w++] = '?'; continue; }
        int l = g_len[t];
        if (l == 0 || l == 255) { out[w++] = ' '; continue; }
        for (int j = 0; j < l && w < max_bytes; j++) out[w++] = g_str[t][j];
    }
    return (int)w;
}

const char *tok_name(void) {
    static char nm[32];
    snprintf(nm, sizeof(nm), "bpe-%d", g_n_merges);
    return g_ready ? nm : "byte-level-fallback";
}
