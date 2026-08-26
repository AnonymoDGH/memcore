#include "tokenizer.h"

#include "config.h"
#include <string.h>

int tok_encode(const unsigned char *text, size_t len, int *out,
               size_t max_tokens) {
    size_t n = len < max_tokens ? len : max_tokens;
    for (size_t i = 0; i < n; i++) out[i] = (int)text[i];
    return (int)n;
}

int tok_decode(const int *tokens, size_t n, unsigned char *out,
               size_t max_bytes) {
    size_t m = n < max_bytes ? n : max_bytes;
    for (size_t i = 0; i < m; i++) {
        int t = tokens[i];
        out[i] = (t >= 0 && t < VOCAB_SIZE) ? (unsigned char)t
                                            : (unsigned char)'?';
    }
    return (int)m;
}

const char *tok_name(void) {
    return "byte-level-256";
}
