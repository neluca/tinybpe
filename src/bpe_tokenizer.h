/*
 * Copyright Yinan Liao. and other contributors. All rights reserved.
 */

#ifndef ULTRA_BPE_TOKENIZER_H
#define ULTRA_BPE_TOKENIZER_H

#include "bpe_common.h"

struct bpe_merges {
    struct avl_tree tree; // merges is a search tree
    void *nodes_mem; // memory of avltree nodes (pair -> ID)
};

struct bpe_token_bytes {
    unsigned char *bytes; // pointer to the bytes corresponding to the Token
    size_t size; // the size of the Token's bytes
};

struct bpe_vocab {
    struct bpe_token_bytes *tokens; // sequential array (ID -> bytes)
    size_t vocab_size;
    unsigned char *bytes_mem; // the total memory occupied by the bytes
};

// build the merges search tree
struct bpe_merges *bpe_merges_build(bpe_pair_t *pairs, size_t len);

// free the memory occupied by merges.
void bpe_merges_free(struct bpe_merges *p);

// search merges to encode the bytes to a sequence of Token IDs.
unsigned long *bpe_encode(size_t *ids_len, const struct bpe_merges *merges, const char *bytes, size_t bytes_size);

// build the dictionary used for BPE decoding.
struct bpe_vocab *bpe_vocab_build(bpe_pair_t *pairs, size_t len);

// free the memory occupied by vocab.
void bpe_vocab_free(struct bpe_vocab *p);

// look up the dictionary to match each Token ID to the bytes.
char *bpe_decode(size_t *bytes_size, const struct bpe_vocab *vocab, const unsigned long *ids, size_t ids_len);

// look up the dictionary to match one Token ID to the bytes.
char *bpe_decode_one(size_t *bytes_size, const struct bpe_vocab *vocab,
                     unsigned long id, unsigned char *cache, unsigned long *cache_size);

#endif  /* ULTRA_BPE_TOKENIZER_H */
