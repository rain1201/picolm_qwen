#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "model.h"
#include <stdint.h>

typedef struct {
    char  **vocab;       /* vocab[i] = string for token i */
    float  *scores;      /* BPE merge scores */
    int     vocab_size;
    int    *sorted_idx;  /* indices sorted by vocab string for binary search */
    uint32_t bos_id;
    uint32_t eos_id;
} tokenizer_t;
typedef int (*encode_ptr)(const tokenizer_t *, const char *, int *, int, int);
typedef const char *(*decode_ptr)(const tokenizer_t *t, int prev_token, int token);
/* Load tokenizer data from GGUF metadata pointers in model.
 * Returns 0 on success. */
int tokenizer_load(tokenizer_t *t, const model_t *m);

/* Encode a text string into token IDs.
 * tokens must have space for at least max_tokens entries.
 * Returns number of tokens produced. */
int tokenizer_encode_llama(const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos);
int tokenizer_encode_qwen(const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos);

/* Decode a single token ID to its string representation.
 * Returns pointer to static/vocab string (do not free). */
const char *tokenizer_decode_llama(const tokenizer_t *t, int prev_token, int token);
const char *tokenizer_decode_qwen(const tokenizer_t *t, int prev_token, int token);
/* Free tokenizer resources */
void tokenizer_free(tokenizer_t *t);

#endif /* TOKENIZER_H */
