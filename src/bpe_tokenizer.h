/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * BPE tokenizer — encoding and decoding.
 *
 * ## Encoding (bytes → token IDs)
 *
 * Greedy lowest-rank-first merging:
 *   1. Convert input bytes to base token IDs (0-255)
 *   2. In each iteration:
 *      a. Look up every adjacent pair in the merges AVL tree
 *      b. Find the pair with the smallest (lowest) merge rank
 *      c. Replace all occurrences of that pair with its merged token ID
 *   3. Repeat until no more merges are possible
 *
 * ## Decoding (token IDs → bytes)
 *
 * Direct O(1) lookup: the flat bpe_vocab array maps each token ID to
 * its byte sequence.  All token bytes are stored in a single contiguous
 * allocation for cache-friendly access.
 *
 * ## Streaming Decode
 *
 * bpe_decode_one() decodes a single token ID at a time, using a 4-byte
 * cache to reassemble UTF-8 sequences that span token boundaries.
 * Complete characters are returned immediately; incomplete bytes are
 * held in the cache for the next call.
 *
 * ## Pure C Portability
 *
 * This module does NOT include <Python.h>.  It is pure C99 and
 * depends only on bpe_common.h and the standard library.
 */

#ifndef SRC_BPE_TOKENIZER_H
#define SRC_BPE_TOKENIZER_H

#include "bpe_common.h"

/* --------------------------------------------------------------------------
 * Merges search tree: maps (left, right) pair → rank.
 *
 * Used during encoding for O(log n) lookup of the merge rank of any
 * adjacent pair of token IDs.
 * -------------------------------------------------------------------------- */
struct bpe_merges {
    struct avl_tree tree;    /* AVL tree root                      */
    void *nodes_mem;         /* contiguous allocation for all nodes */
};

/* --------------------------------------------------------------------------
 * Individual token entry in the flat vocabulary.
 * -------------------------------------------------------------------------- */
struct bpe_token_bytes {
    unsigned char *bytes;    /* pointer into vocab->bytes_mem */
    size_t size;             /* number of bytes for this token */
};

/* --------------------------------------------------------------------------
 * Flat vocabulary: (token ID) → bytes.
 *
 * tokens[] is indexed by token ID (0 to vocab_size-1).  All byte
 * sequences are packed into a single bytes_mem allocation for
 * cache efficiency.
 * -------------------------------------------------------------------------- */
struct bpe_vocab {
    struct bpe_token_bytes *tokens;  /* array indexed by token ID    */
    size_t vocab_size;               /* 256 + n_merges               */
    unsigned char *bytes_mem;        /* single allocation for bytes  */
};

/* --------------------------------------------------------------------------
 * Build the merges search tree from an array of merge pairs.
 *
 * Each pair's rank is 256 + its index in the array.  Returns NULL on
 * allocation failure (MemoryError already set by bpe_malloc).
 * -------------------------------------------------------------------------- */
struct bpe_merges *bpe_merges_build(bpe_pair_t *pairs, size_t len);

/* --------------------------------------------------------------------------
 * Free a merges search tree.  Safe to call with NULL.
 * -------------------------------------------------------------------------- */
void bpe_merges_free(struct bpe_merges *m);

/* --------------------------------------------------------------------------
 * Encode a byte sequence into BPE token IDs.
 *
 * Uses greedy lowest-rank-first merging.  The returned array must be
 * freed by the caller via bpe_free().  On allocation failure, returns
 * NULL with *ids_len = 0.
 *
 * Parameters:
 *   ids_len    — [out] number of token IDs produced
 *   merges     — the merges search tree
 *   bytes      — input byte sequence
 *   bytes_size — number of input bytes
 *
 * Returns: dynamically allocated array of unsigned long token IDs.
 *          Caller must bpe_free() it.
 * -------------------------------------------------------------------------- */
unsigned long *bpe_encode(size_t *ids_len, const struct bpe_merges *merges,
                          const char *bytes, size_t bytes_size);

/* --------------------------------------------------------------------------
 * Build the flat vocabulary from merge pairs.
 *
 * Creates the mapping from token IDs (0 through 255 + len) to their
 * byte sequences by iteratively concatenating the component bytes of
 * each merge pair.  Returns NULL on allocation failure.
 * -------------------------------------------------------------------------- */
struct bpe_vocab *bpe_vocab_build(bpe_pair_t *pairs, size_t len);

/* --------------------------------------------------------------------------
 * Free a vocabulary.  Safe to call with NULL.
 * -------------------------------------------------------------------------- */
void bpe_vocab_free(struct bpe_vocab *v);

/* --------------------------------------------------------------------------
 * Decode a list of token IDs to bytes (batch mode).
 *
 * Concatenates the byte sequences of all token IDs in order.
 * Returns NULL with *bytes_size = 0 if any ID is out of range.
 * Caller must bpe_free() the returned buffer.
 * -------------------------------------------------------------------------- */
char *bpe_decode(size_t *bytes_size, const struct bpe_vocab *vocab,
                 const unsigned long *ids, size_t ids_len);

/* --------------------------------------------------------------------------
 * Decode a single token ID (streaming mode).
 *
 * Appends the token's bytes to the internal cache, then extracts and
 * returns complete leading UTF-8 characters.  Incomplete multi-byte
 * sequences are retained in the cache for the next call.
 *
 * If the cache contains an invalid UTF-8 start byte (continuation or
 * 0xF5-0xFF), the byte is flushed as a raw single-byte fragment to
 * ensure forward progress.
 *
 * Parameters:
 *   bytes_size — [out] number of bytes returned (may be 0 if no
 *                       complete character was formed yet)
 *   vocab      — the vocabulary
 *   id         — the token ID to decode
 *   cache      — 4-byte internal buffer for partial UTF-8 sequences
 *   cache_size — [in/out] number of valid bytes currently in the cache
 *
 * Returns: allocated buffer (caller must bpe_free).  *bytes_size may
 *          be 0 if no complete character could be formed yet.
 * -------------------------------------------------------------------------- */
char *bpe_decode_one(size_t *bytes_size, const struct bpe_vocab *vocab,
                     unsigned long id, unsigned char *cache,
                     unsigned long *cache_size);

#endif  /* SRC_BPE_TOKENIZER_H */
