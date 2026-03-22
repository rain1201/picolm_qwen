#ifndef GGUF_KEYS_H
#define GGUF_KEYS_H

/* 基础架构键 */
#define GKEY_EMBED_LEN      "embedding_length"
#define GKEY_FFN_LEN        "feed_forward_length"
#define GKEY_HEAD_COUNT     "attention.head_count"
#define GKEY_HEAD_COUNT_KV  "attention.head_count_kv"
#define GKEY_BLOCK_COUNT    "block_count"
#define GKEY_CONTEXT_LEN    "context_length"
#define GKEY_ROPE_FREQ      "rope.freq_base"
#define GKEY_QK_NORM        "attention.key_length" // Qwen 3 的头维度标识

/* 通用键 */
#define GKEY_ALIGNMENT      "general.alignment"
#define GKEY_VOCAB_SIZE     "llama.vocab_size"
#define GKEY_BOS_ID         "tokenizer.ggml.bos_token_id"
#define GKEY_EOS_ID         "tokenizer.ggml.eos_token_id"

#endif