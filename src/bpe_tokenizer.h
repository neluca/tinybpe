/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * BPE tokenizer — encoding (bytes → token IDs) and decoding (token IDs → bytes).
 *
 * Two core data structures:
 *
 *   bpe_merges — AVL tree mapping (left, right) pair → rank.
 *                Used for encoding: given adjacent token IDs, find the
 *                lowest-rank merge that combines them.
 *
 *   bpe_vocab  — flat array mapping token ID → byte sequence.
 *                Used for decoding: given a token ID, return its bytes.
 *                All token bytes are stored in a single contiguous
 *                allocation for cache-friendly decoding.
 *
 * Streaming decode via bpe_decode_one() decodes one token at a time,
 * using a 4-byte cache to handle UTF-8 character boundaries.
 */

#ifndef SRC_BPE_TOKENIZER_H
#define SRC_BPE_TOKENIZER_H

#include "bpe_common.h"

/* Search tree: (pair) → rank, used for encoding (O(log n) lookup). */
struct bpe_merges {
    struct avl_tree tree;    // AVL tree root
    void *nodes_mem;         // contiguous memory for all tree nodes
};

/* Individual token entry in the vocabulary. */
struct bpe_token_bytes {
    unsigned char *bytes;    // pointer into vocab->bytes_mem
    size_t size;             // number of bytes for this token
};

/* Flat vocabulary: (token ID) → bytes, used for decoding (O(1) lookup). */
struct bpe_vocab {
    struct bpe_token_bytes *tokens;  // sequential array indexed by token ID
    size_t vocab_size;               // total number of tokens (256 + n_merges)
    unsigned char *bytes_mem;        // single allocation holding all token bytes
};

/*
 * Build the merges search tree from an array of merge pairs.
 *
 * Each pair's rank is 256 + its index in the array.
 * Returns NULL on allocation failure.
 */
struct bpe_merges *bpe_merges_build(bpe_pair_t *pairs, size_t len);

/*
 * Free the merges search tree.
 * Safe to call with NULL.
 */
void bpe_merges_free(struct bpe_merges *p);

/*
 * Encode a byte sequence into BPE token IDs.
 *
 * Uses greedy lowest-rank-first merging:
 * repeatedly finds the adjacent pair with the smallest merge rank
 * and replaces it, until no more merges are possible.
 *
 * Returns: dynamically allocated array of token IDs (caller must bpe_free).
 *          *ids_len is set to the array length.
 *          Returns NULL on allocation failure.
 */
unsigned long *bpe_encode(size_t *ids_len, const struct bpe_merges *merges, const char *bytes, size_t bytes_size);

/*
 * Build the vocabulary from merge pairs.
 *
 * Creates a flat mapping from token IDs (0 through 255 + len) to
 * their byte sequences by iteratively concatenating merge pairs.
 * Returns NULL on allocation failure.
 */
struct bpe_vocab *bpe_vocab_build(bpe_pair_t *pairs, size_t len);

/*
 * Free the vocabulary.
 * Safe to call with NULL.
 */
void bpe_vocab_free(struct bpe_vocab *p);

/*
 * Decode a list of token IDs to bytes (batch mode).
 *
 * Concatenates the byte sequences of all token IDs in order.
 * Returns NULL (with *bytes_size = 0) if any ID is out of range.
 * Caller must bpe_free() the returned buffer.
 */
char *bpe_decode(size_t *bytes_size, const struct bpe_vocab *vocab, const unsigned long *ids, size_t ids_len);

/*
 * Decode a single token ID (streaming mode).
 *
 * Appends the token's bytes to the internal cache and extracts leading
 * complete UTF-8 characters. Returns partial bytes in the cache for
 * the next call. Handles multi-byte UTF-8 sequences that span token
 * boundaries.
 *
 * Parameters:
 *   bytes_size — [out] number of bytes returned in the result
 *   vocab      — the vocabulary
 *   id         — the token ID to decode
 *   cache      — 4-byte internal buffer for partial UTF-8 sequences
 *   cache_size — [in/out] number of bytes currently in the cache
 *
 * Returns: allocated buffer (caller must bpe_free). *bytes_size may
 *          be 0 if no complete character could be formed yet.
 */
char *bpe_decode_one(size_t *bytes_size, const struct bpe_vocab *vocab,
                     unsigned long id, unsigned char *cache, unsigned long *cache_size);

#endif  /* SRC_BPE_TOKENIZER_H */
