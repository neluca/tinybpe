/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * BPE tokenizer — encoding and decoding implementations.
 *
 * ## Encoding (text → token IDs)
 *
 * Encoding uses a greedy lowest-rank-first algorithm:
 *   1. Convert input bytes to a sequence of base token IDs (0-255)
 *   2. In each iteration:
 *      a. Look up every adjacent pair in the merges AVL tree
 *      b. Find the pair with the lowest merge rank
 *      c. Replace all occurrences of that pair with its merged token ID
 *   3. Repeat until no more merges can be applied
 *
 * This greedy approach ensures deterministic, reproducible encoding:
 * the same merges always produce the same token IDs for the same input.
 *
 * ## Decoding (token IDs → bytes)
 *
 * Decoding is a direct lookup: each token ID maps to a byte sequence
 * in the flat vocab array, constructed by iteratively applying merges
 * to the 256 base byte tokens.
 *
 * ## Streaming Decode
 *
 * bpe_decode_one() decodes a single token ID at a time, using a 4-byte
 * cache to handle partial UTF-8 sequences that span token boundaries.
 * When a complete UTF-8 character is accumulated, it is returned to
 * the caller; incomplete bytes are held in the cache for the next call.
 *
 * ## Data Structures
 *
 *   bpe_merges — AVL tree mapping (left, right) pair → rank
 *     Used for encoding lookups: given two adjacent token IDs,
 *     find the lowest-rank merge that produces a larger token.
 *
 *   bpe_vocab — flat array mapping token ID → byte sequence
 *     Used for decoding: given a token ID, return its byte representation.
 *     Bytes for all tokens are stored in a single contiguous allocation
 *     (bytes_mem) to minimize malloc overhead.
 */

#include "bpe_tokenizer.h"
#include <string.h>

/* AVL tree node for the merges search tree (pair → rank). */
struct bpe_merges_node {
    struct avl_node node;
    bpe_pair_t pair;
    unsigned long rank;
};

/* Temporary storage for pair-to-rank lookup during encoding. */
struct bpe_pair_stats {
    bpe_pair_t pair;
    unsigned long merges_rank; // ULONG_MAX (= -1) means "no merge found"
};

static int merges_cmp_func(struct avl_node *a, struct avl_node *b) {
    struct bpe_merges_node *n1, *n2;

    n1 = _get_entry(a, struct bpe_merges_node, node);
    n2 = _get_entry(b, struct bpe_merges_node, node);

    return bpe_pair_cmp(&n1->pair, &n2->pair);
}

/*
 * Build the merges search tree from an array of merge pairs.
 *
 * The tree maps each (left, right) pair to its rank (256 + index).
 * This allows O(log n) lookup during encoding to find the merge rank
 * of any adjacent pair of token IDs.
 */
struct bpe_merges *bpe_merges_build(bpe_pair_t *pairs, size_t len) {
    struct bpe_merges *merges = bpe_malloc(sizeof(struct bpe_merges));

    avl_init(&merges->tree);

    struct bpe_merges_node *node_buf = bpe_malloc(len * sizeof(struct bpe_merges_node));
    merges->nodes_mem = node_buf;

    for (size_t i = 0; i < len; i++) {
        node_buf[i].pair = pairs[i];
        node_buf[i].rank = (unsigned long) (256 + i);

        avl_insert(&merges->tree, &node_buf[i].node, merges_cmp_func);
    }

    return merges;
}

/*
 * Free the merges search tree and all associated memory.
 * Safe to call with NULL.
 */
void bpe_merges_free(struct bpe_merges *p) {
    if (p) {
        bpe_free(p->nodes_mem);
        p->nodes_mem = NULL;
        bpe_free(p);
    }
}

/*
 * Encode a byte sequence into BPE token IDs using greedy lowest-rank merging.
 *
 * Algorithm:
 *   while (sequence length > 1):
 *     1. For each adjacent pair, look up its merge rank in the merges tree
 *     2. Find the pair with the lowest rank
 *     3. If no pair has a valid rank, stop (encoding is complete)
 *     4. Replace all occurrences of the lowest-rank pair with its merged ID
 *
 * The output is written to buf_ids[0..*ids_len-1].
 * The caller must bpe_free() the returned buffer.
 */
unsigned long *bpe_encode(size_t *ids_len, const struct bpe_merges *merges, const char *bytes, size_t bytes_size) {
    // Guard against overflow
    if (bytes_size > SIZE_MAX / sizeof(unsigned long)) {
        *ids_len = 0;
        return NULL;
    }
    unsigned long *buf_ids = bpe_malloc(bytes_size * sizeof(unsigned long));

    // Initialize: each byte becomes a base token ID (0-255)
    for (size_t i = 0; i < bytes_size; i++) {
        buf_ids[i] = (unsigned long) ((unsigned char) bytes[i]);
    }

    size_t len = bytes_size;
    struct bpe_pair_stats *stats = bpe_malloc((len - 1) * sizeof(struct bpe_pair_stats));

    struct bpe_merges_node lookup;
    while (len > 1) {
        // Phase 1: look up the merge rank for every adjacent pair
        for (size_t i = 0; i < len - 1; i++) {
            lookup.pair.left = buf_ids[i];
            lookup.pair.right = buf_ids[i + 1];
            stats[i].pair = lookup.pair;

            struct avl_node *_node = avl_search(&merges->tree, &lookup.node, merges_cmp_func);
            if (_node) {
                struct bpe_merges_node *_n = _get_entry(_node, struct bpe_merges_node, node);
                stats[i].merges_rank = _n->rank;
            }
            else {
                stats[i].merges_rank = (unsigned long) (-1); // sentinel: no merge
            }
        }

        // Phase 2: find the pair with the lowest merge rank
        struct bpe_pair_stats *_min = &stats[0];
        for (size_t i = 1; i < len - 1; i++) {
            if (stats[i].merges_rank < _min->merges_rank) {
                _min = &stats[i];
            }
        }

        if (_min->merges_rank == (unsigned long) (-1)) {
            break; // no more merges possible
        }
        else {
            // Phase 3: compact the sequence by merging the winning pair
            size_t new_ids_i = 0;
            for (size_t i = 0; i < len; i++) {
                if (buf_ids[i] == _min->pair.left && i < len - 1 && buf_ids[i + 1] == _min->pair.right) {
                    buf_ids[new_ids_i++] = _min->merges_rank;
                    i++; // skip the right half of the pair
                }
                else {
                    buf_ids[new_ids_i++] = buf_ids[i];  // new_ids_i always <= i, safe in-place
                }
            }

            len = new_ids_i;
        }
    }

    bpe_free(stats);

    *ids_len = len;
    return buf_ids;
}

/*
 * Build the vocabulary (token ID → bytes mapping) from merge pairs.
 *
 * The vocabulary starts with the 256 single-byte tokens (IDs 0-255).
 * For each merge pair (left, right), the new token's bytes are the
 * concatenation of left's bytes and right's bytes.
 *
 * Memory layout: all token byte sequences are stored in a single
 * contiguous allocation (bytes_mem) for cache efficiency.
 */
struct bpe_vocab *bpe_vocab_build(bpe_pair_t *pairs, size_t len) {
    struct bpe_vocab *vocab = bpe_malloc(sizeof(struct bpe_vocab));
    vocab->vocab_size = 256 + len;

    size_t total_bytes_size = 256;
    // Track the byte size of each merged token for total size calculation
    size_t *id_bytes_size_buf = bpe_malloc(len * sizeof(size_t));

    // First pass: calculate total memory needed for all token byte sequences
    for (size_t i = 0; i < len; i++) {
        size_t _size = 0;
        if (pairs[i].left < 256) {
            _size += 1;
        }
        else {
            _size += id_bytes_size_buf[pairs[i].left - 256];
        }

        if (pairs[i].right < 256) {
            _size += 1;
        }
        else {
            _size += id_bytes_size_buf[pairs[i].right - 256];
        }

        id_bytes_size_buf[i] = _size;
        total_bytes_size += _size;
    }

    vocab->bytes_mem = bpe_malloc(total_bytes_size);
    vocab->tokens = bpe_malloc(vocab->vocab_size * sizeof(struct bpe_token_bytes));

    // Initialize base tokens: IDs 0-255 map to single bytes
    for (size_t i = 0; i < 256; i++) {
        vocab->bytes_mem[i] = (unsigned char) i;
        vocab->tokens[i].bytes = &vocab->bytes_mem[i];
        vocab->tokens[i].size = 1;
    }

    // Build merged tokens: concatenate left and right byte sequences
    unsigned char *bytes_mem_p = vocab->bytes_mem + 256;
    for (size_t i = 0; i < len; i++) {

        memcpy(bytes_mem_p, vocab->tokens[pairs[i].left].bytes, vocab->tokens[pairs[i].left].size);
        size_t _size = vocab->tokens[pairs[i].left].size;
        memcpy(bytes_mem_p + _size, vocab->tokens[pairs[i].right].bytes, vocab->tokens[pairs[i].right].size);

        vocab->tokens[i + 256].bytes = bytes_mem_p;
        vocab->tokens[i + 256].size = id_bytes_size_buf[i];

        bytes_mem_p += id_bytes_size_buf[i];
    }

    bpe_free(id_bytes_size_buf);

    return vocab;
}

/*
 * Free the vocabulary and all associated memory.
 * Safe to call with NULL.
 */
void bpe_vocab_free(struct bpe_vocab *p) {
    if (p) {
        bpe_free(p->tokens);
        p->tokens = NULL;
        bpe_free(p->bytes_mem);
        p->bytes_mem = NULL;
        bpe_free(p);
    }
}

/*
 * Decode a list of token IDs back into bytes (batch mode).
 *
 * Concatenates the byte sequences for each token ID in order.
 * Returns NULL (with *bytes_size = 0) if any token ID is out of range.
 * The caller must bpe_free() the returned buffer.
 */
char *bpe_decode(size_t *bytes_size, const struct bpe_vocab *vocab, const unsigned long *ids, size_t ids_len) {
    // Calculate total output size
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

    // Concatenate token byte sequences
    for (size_t i = 0; i < ids_len; i++) {
        memcpy(p, vocab->tokens[ids[i]].bytes, vocab->tokens[ids[i]].size);
        p += vocab->tokens[ids[i]].size;
    }

    return buf_bytes;
}

/*
 * Decode a single token ID for streaming decode.
 *
 * Appends the token's bytes to an internal cache, then extracts and
 * returns the leading complete UTF-8 character(s). Incomplete multi-byte
 * sequences are left in the cache for the next call.
 *
 * Parameters:
 *   bytes_size — [out] number of bytes returned
 *   vocab      — the vocabulary
 *   id         — the token ID to decode
 *   cache      — 4-byte internal cache for partial UTF-8 sequences
 *   cache_size — [in/out] number of valid bytes currently in the cache
 *
 * Returns: allocated buffer containing the decoded bytes (caller must free),
 *          with *bytes_size = 0 if no complete character could be formed yet.
 */
char *bpe_decode_one(size_t *bytes_size, const struct bpe_vocab *vocab,
                     unsigned long id, unsigned char *cache, unsigned long *cache_size) {
    size_t buf_size = vocab->tokens[id].size + (size_t) *cache_size;
    unsigned char *buf_bytes = bpe_malloc(buf_size);
    unsigned char *p = buf_bytes;
    if (*cache_size) {
        memcpy(p, cache, (size_t) *cache_size);
        p += (size_t) *cache_size;
    }

    memcpy(p, vocab->tokens[id].bytes, vocab->tokens[id].size);
    size_t utf8_size = 0;

    // Walk through the buffer, consuming complete UTF-8 characters.
    // Continuation bytes and invalid bytes are treated as 1-byte
    // fragments to ensure forward progress.
    size_t i = bpe_utf8_length_from_head(buf_bytes[0]);
    if (i == 0) i = 1; // treat continuation/invalid byte as 1-byte fragment
    while (i <= buf_size) {
        utf8_size = i;
        if (i == buf_size) {
            break;
        }

        size_t next_len = bpe_utf8_length_from_head(buf_bytes[i]);
        i += (next_len == 0) ? 1 : next_len;
    }

    *bytes_size = utf8_size;
    *cache_size = (unsigned long) (buf_size - utf8_size);

    if (*cache_size) {
        // Incomplete bytes remain — save for next call
        memcpy(cache, buf_bytes + utf8_size, (size_t) *cache_size);
    }

    return (char *) buf_bytes;
}
