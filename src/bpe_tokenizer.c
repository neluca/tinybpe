/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * BPE tokenizer — encoding and decoding implementations (pure C).
 *
 * ## Encoding (bytes → token IDs)
 *
 * Greedy lowest-rank-first algorithm:
 *   1. Each byte becomes a base token ID (0-255)
 *   2. Each iteration:
 *      a. Look up the merge rank of every adjacent pair in the AVL tree
 *      b. Find the pair with the smallest (lowest) rank
 *      c. If no pair has a valid rank, stop — encoding complete
 *      d. Compact the sequence by replacing all occurrences of the
 *         lowest-rank pair with its merged token ID
 *   3. Return the final token ID sequence
 *
 * This greedy approach is deterministic: the same merges list always
 * produces the same token IDs for the same input.
 *
 * ## Decoding (token IDs → bytes)
 *
 * Batch decode: concatenate each token's byte sequence.
 * Streaming decode: process one token at a time through a 4-byte cache
 * that reassembles UTF-8 characters spanning token boundaries.
 *
 * ## Data Structures
 *
 *   bpe_merges     — AVL tree: (left, right) pair → rank
 *   bpe_vocab      — flat array: token ID → byte sequence
 *   bytes_cache[4] — streaming decode reassembly buffer
 */

#include "bpe_tokenizer.h"
#include <string.h>

/* --------------------------------------------------------------------------
 * AVL tree node for the merges search tree.  Maps a (left, right) pair
 * to its merge rank (256 + index).
 * -------------------------------------------------------------------------- */
struct bpe_merges_node {
    struct avl_node node;
    bpe_pair_t pair;
    unsigned long rank;
};

/* --------------------------------------------------------------------------
 * Temporary record for pair-to-rank lookup during encoding.
 * ULONG_MAX (= ~0UL) serves as the "no merge found" sentinel.
 * -------------------------------------------------------------------------- */
struct bpe_pair_stats {
    bpe_pair_t pair;
    unsigned long merges_rank;
};

static int merges_cmp_func(struct avl_node *a, struct avl_node *b) {
    struct bpe_merges_node *n1, *n2;

    n1 = _get_entry(a, struct bpe_merges_node, node);
    n2 = _get_entry(b, struct bpe_merges_node, node);

    return bpe_pair_cmp(&n1->pair, &n2->pair);
}

/* --------------------------------------------------------------------------
 * Build the merges search tree.
 *
 * Each pair gets rank = 256 + index.  All nodes are allocated in a
 * single contiguous block for cache efficiency.
 * -------------------------------------------------------------------------- */
struct bpe_merges *bpe_merges_build(bpe_pair_t *pairs, size_t len) {
    struct bpe_merges *merges = bpe_malloc(sizeof(struct bpe_merges));

    avl_init(&merges->tree);

    struct bpe_merges_node *node_buf =
        bpe_malloc(len * sizeof(struct bpe_merges_node));
    merges->nodes_mem = node_buf;

    for (size_t i = 0; i < len; i++) {
        node_buf[i].pair = pairs[i];
        node_buf[i].rank = (unsigned long)(256 + i);

        avl_insert(&merges->tree, &node_buf[i].node, merges_cmp_func);
    }

    return merges;
}

/* --------------------------------------------------------------------------
 * Free the merges search tree.
 * -------------------------------------------------------------------------- */
void bpe_merges_free(struct bpe_merges *m) {
    if (m) {
        bpe_free(m->nodes_mem);
        m->nodes_mem = NULL;
        bpe_free(m);
    }
}

/* --------------------------------------------------------------------------
 * Encode bytes → token IDs via greedy lowest-rank-first merging.
 *
 * The output buffer starts at the same size as the input (worst case:
 * no merges applied).  Each merge iteration compacts the buffer
 * in-place (new_ids_i ≤ current index, so no overwrite risk).
 *
 * The stats array is allocated once and reused across iterations.
 * Its size shrinks with the sequence length.
 * -------------------------------------------------------------------------- */
unsigned long *bpe_encode(size_t *ids_len, const struct bpe_merges *merges,
                          const char *bytes, size_t bytes_size) {
    /* Guard against overflow */
    if (bytes_size > SIZE_MAX / sizeof(unsigned long)) {
        *ids_len = 0;
        return NULL;
    }
    unsigned long *buf_ids = bpe_malloc(bytes_size * sizeof(unsigned long));

    /* Initialize: each byte → base token ID (0-255) */
    for (size_t i = 0; i < bytes_size; i++) {
        buf_ids[i] = (unsigned long)((unsigned char)bytes[i]);
    }

    size_t len = bytes_size;
    struct bpe_pair_stats *stats =
        bpe_malloc((len - 1) * sizeof(struct bpe_pair_stats));

    struct bpe_merges_node lookup;
    while (len > 1) {
        /* Phase 1: look up the merge rank for every adjacent pair */
        for (size_t i = 0; i < len - 1; i++) {
            lookup.pair.left = buf_ids[i];
            lookup.pair.right = buf_ids[i + 1];
            stats[i].pair = lookup.pair;

            struct avl_node *_node =
                avl_search(&merges->tree, &lookup.node, merges_cmp_func);
            if (_node) {
                struct bpe_merges_node *_n =
                    _get_entry(_node, struct bpe_merges_node, node);
                stats[i].merges_rank = _n->rank;
            }
            else {
                stats[i].merges_rank = (unsigned long)(-1); /* no merge */
            }
        }

        /* Phase 2: find the pair with the lowest merge rank */
        struct bpe_pair_stats *_min = &stats[0];
        for (size_t i = 1; i < len - 1; i++) {
            if (stats[i].merges_rank < _min->merges_rank) {
                _min = &stats[i];
            }
        }

        if (_min->merges_rank == (unsigned long)(-1)) {
            break; /* no more merges possible */
        }

        /* Phase 3: compact the sequence by merging the winning pair.
         * new_ids_i ≤ i always holds, so in-place is safe. */
        size_t new_ids_i = 0;
        for (size_t i = 0; i < len; i++) {
            if (buf_ids[i] == _min->pair.left
                && i < len - 1
                && buf_ids[i + 1] == _min->pair.right) {
                buf_ids[new_ids_i++] = _min->merges_rank;
                i++; /* skip right half */
            }
            else {
                buf_ids[new_ids_i++] = buf_ids[i];
            }
        }

        len = new_ids_i;
    }

    bpe_free(stats);

    *ids_len = len;
    return buf_ids;
}

/* --------------------------------------------------------------------------
 * Build the flat vocabulary from merge pairs.
 *
 * Starts with the 256 single-byte tokens (IDs 0-255).  For each merge
 * pair (left, right), the new token's bytes are the concatenation of
 * the left token's bytes and the right token's bytes.
 *
 * Two-pass approach:
 *   Pass 1: calculate total memory needed for all token byte sequences
 *   Pass 2: fill the single contiguous bytes_mem allocation
 * -------------------------------------------------------------------------- */
struct bpe_vocab *bpe_vocab_build(bpe_pair_t *pairs, size_t len) {
    struct bpe_vocab *vocab = bpe_malloc(sizeof(struct bpe_vocab));
    vocab->vocab_size = 256 + len;

    /* Pass 1: calculate sizes */
    size_t total_bytes_size = 256;
    size_t *id_bytes_size_buf = bpe_malloc(len * sizeof(size_t));

    for (size_t i = 0; i < len; i++) {
        size_t _size = 0;
        _size += (pairs[i].left < 256)
                     ? 1
                     : id_bytes_size_buf[pairs[i].left - 256];
        _size += (pairs[i].right < 256)
                     ? 1
                     : id_bytes_size_buf[pairs[i].right - 256];

        id_bytes_size_buf[i] = _size;
        total_bytes_size += _size;
    }

    vocab->bytes_mem = bpe_malloc(total_bytes_size);
    vocab->tokens = bpe_malloc(vocab->vocab_size * sizeof(struct bpe_token_bytes));

    /* Initialize base tokens: IDs 0-255 map to single bytes */
    for (size_t i = 0; i < 256; i++) {
        vocab->bytes_mem[i] = (unsigned char)i;
        vocab->tokens[i].bytes = &vocab->bytes_mem[i];
        vocab->tokens[i].size = 1;
    }

    /* Pass 2: build merged tokens by concatenating left + right bytes */
    unsigned char *bytes_mem_p = vocab->bytes_mem + 256;
    for (size_t i = 0; i < len; i++) {
        memcpy(bytes_mem_p,
               vocab->tokens[pairs[i].left].bytes,
               vocab->tokens[pairs[i].left].size);
        size_t left_size = vocab->tokens[pairs[i].left].size;
        memcpy(bytes_mem_p + left_size,
               vocab->tokens[pairs[i].right].bytes,
               vocab->tokens[pairs[i].right].size);

        vocab->tokens[i + 256].bytes = bytes_mem_p;
        vocab->tokens[i + 256].size = id_bytes_size_buf[i];

        bytes_mem_p += id_bytes_size_buf[i];
    }

    bpe_free(id_bytes_size_buf);
    return vocab;
}

/* --------------------------------------------------------------------------
 * Free the vocabulary.
 * -------------------------------------------------------------------------- */
void bpe_vocab_free(struct bpe_vocab *v) {
    if (v) {
        bpe_free(v->tokens);
        v->tokens = NULL;
        bpe_free(v->bytes_mem);
        v->bytes_mem = NULL;
        bpe_free(v);
    }
}

/* --------------------------------------------------------------------------
 * Batch decode: concatenate byte sequences for all token IDs.
 *
 * Two-pass: first sum the total output size, then copy bytes.
 * Returns NULL (with *bytes_size = 0) if any token ID is out of range.
 * -------------------------------------------------------------------------- */
char *bpe_decode(size_t *bytes_size, const struct bpe_vocab *vocab,
                 const unsigned long *ids, size_t ids_len) {
    /* Calculate total output size */
    size_t buf_size = 0;
    for (size_t i = 0; i < ids_len; i++) {
        if (ids[i] >= vocab->vocab_size) {
            *bytes_size = 0;
            return NULL;
        }
        buf_size += vocab->tokens[ids[i]].size;
    }

    *bytes_size = buf_size;
    char *buf_bytes = bpe_malloc(buf_size);
    char *p = buf_bytes;

    /* Concatenate token byte sequences */
    for (size_t i = 0; i < ids_len; i++) {
        memcpy(p, vocab->tokens[ids[i]].bytes, vocab->tokens[ids[i]].size);
        p += vocab->tokens[ids[i]].size;
    }

    return buf_bytes;
}

/* --------------------------------------------------------------------------
 * Streaming decode: decode one token ID at a time through a cache.
 *
 * Algorithm:
 *   1. Append the token's bytes to the cache
 *   2. Walk forward through the buffer, consuming complete UTF-8 chars
 *   3. Return the leading complete characters
 *   4. Retain any partial multi-byte sequence in the cache
 *
 * If the cache starts with an invalid UTF-8 lead byte (continuation
 * byte, 0xF5-0xFF), it is flushed as a raw single byte to ensure
 * forward progress — this handles edge cases with malformed input.
 * -------------------------------------------------------------------------- */
char *bpe_decode_one(size_t *bytes_size, const struct bpe_vocab *vocab,
                     unsigned long id, unsigned char *cache,
                     unsigned long *cache_size) {
    /* Validate cache: if it starts with an invalid UTF-8 lead byte,
     * flush it as a raw byte to ensure forward progress */
    if (*cache_size && !bpe_utf8_length_from_head(cache[0])) {
        *cache_size = 0;
    }

    size_t buf_size = vocab->tokens[id].size + (size_t)(*cache_size);
    unsigned char *buf_bytes = bpe_malloc(buf_size);
    unsigned char *p = buf_bytes;

    /* Restore cached partial bytes */
    if (*cache_size) {
        memcpy(p, cache, (size_t)(*cache_size));
        p += (size_t)(*cache_size);
    }

    /* Append the new token's bytes */
    memcpy(p, vocab->tokens[id].bytes, vocab->tokens[id].size);

    /* Walk through the buffer consuming complete UTF-8 characters.
     * If a lead byte is invalid (continuation / >0xF4), treat it as
     * a 1-byte fragment so we don't stall. */
    size_t utf8_size = 0;
    size_t i = bpe_utf8_length_from_head(buf_bytes[0]);
    if (i == 0) i = 1;

    while (i <= buf_size) {
        utf8_size = i;
        if (i == buf_size) {
            break;
        }

        size_t next_len = bpe_utf8_length_from_head(buf_bytes[i]);
        i += (next_len == 0) ? 1 : next_len;
    }

    *bytes_size = utf8_size;
    *cache_size = (unsigned long)(buf_size - utf8_size);

    /* Save incomplete tail bytes for next call */
    if (*cache_size) {
        memcpy(cache, buf_bytes + utf8_size, (size_t)(*cache_size));
    }

    return (char *)buf_bytes;
}
