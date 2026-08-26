#ifndef MEMCORE_TOKENIZER_H
#define MEMCORE_TOKENIZER_H

#include <stddef.h>

int tok_ensure(const char *corpus_path, const char *model_path);
int tok_load(const char *model_path);
int tok_ready(void);
int tok_vocab(void);

int tok_encode(const unsigned char *text, size_t len, int *out,
               size_t max_tokens);
int tok_decode(const int *tokens, size_t n, unsigned char *out,
               size_t max_bytes);
const char *tok_name(void);

#endif
