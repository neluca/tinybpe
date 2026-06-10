/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * BPE training algorithm implementation.
 *
 * ## Algorithm Overview
 *
 * The BPE trainer learns a vocabulary of merge pairs from training data.
 * Starting from the 256 base byte tokens, each training step:
 *
 *   1. Scans all adjacent pairs across all training pieces
 *   2. Builds an AVL tree counting pair frequencies
 *   3. Finds the most frequent pair
 *   4. Replaces all occurrences of that pair with a new token ID
 *   5. The new token ID becomes available for future merges
 *
 * Training completes when no more pairs are available or when the
 * desired vocabulary size is reached.
 *
 * ## Data Structures
 *
 *   bpe_train_ctx_t — training context holding:
 *     - pieces[]: array of bpe_piece_t, each being a sequence of token IDs
 *       representing one chunk of the training corpus
 *     - rank: the next token ID to assign (starts at 256)
 *
 *   bpe_pair_stats_node — AVL tree node tracking:
 *     - pair: the (left, right) token ID pair
 *     - count: how many times this pair appears
 */

#include "bpe_trainer.h"

/* AVL tree node for pair frequency statistics during training. */
struct bpe_pair_stats_node {
    struct avl_node node;
    bpe_pair_t pair;
    size_t count;
};

/*
 * Replace all occurrences of a pair in the training pieces with a new token ID.
 *
 * Operates in-place on each piece's ids[] array. After replacement, the
 * piece length shrinks because each pair becomes a single merged ID.
 *
 * Example: piece ids = [h, e, l, l, o], pair = (h, e), id = 256
 * Result: pieces ids = [256, l, l, o], len = 4
 */
static inline void merge_piece(bpe_piece_t *pieces, size_t pieces_len, const bpe_pair_t *pair, unsigned long id) {
    for (size_t i = 0; i < pieces_len; i++) {
        if (pieces[i].len > 1) {
            unsigned long *p_ids = pieces[i].ids;

            size_t new_ids_i = 0;
            for (size_t j = 0; j < pieces[i].len; j++) {
                if (p_ids[j] == pair->left && j < pieces[i].len - 1 && p_ids[j + 1] == pair->right) {
                    p_ids[new_ids_i++] = id;
                    j++; // skip the right half of the pair
                }
                else {
                    p_ids[new_ids_i++] = p_ids[j];
                }
            }

            pieces[i].len = new_ids_i;
        }
    }
}

static int pair_stat_cmp_func(struct avl_node *a, struct avl_node *b) {
    struct bpe_pair_stats_node *n1, *n2;

    n1 = _get_entry(a, struct bpe_pair_stats_node, node);
    n2 = _get_entry(b, struct bpe_pair_stats_node, node);

    return bpe_pair_cmp(&n1->pair, &n2->pair);
}

/*
 * Find the most frequent adjacent pair across all training pieces.
 *
 * This is the core of the BPE training step:
 *   1. Iterate over all pieces, collecting every adjacent pair
 *   2. Count pair frequencies using an AVL tree (log n lookup per pair)
 *   3. Find the pair with the highest count
 *   4. Apply the merge (replace all occurrences of the winning pair)
 *   5. Return the pair, its new rank, and its frequency
 *
 * Returns the frequency count (> 0) on success, or 0 if no pairs are available.
 */
unsigned long bpe_get_max_count_pair(bpe_pair_t *pair, bpe_train_ctx_t *ctx) {
    struct avl_tree tree;
    avl_init(&tree);

    // Calculate total number of adjacent pairs across all pieces
    size_t stats_len = 0;
    for (size_t i = 0; i < ctx->pieces_len; i++) {
        stats_len += ctx->pieces[i].len - 1;
    }

    if (stats_len > SIZE_MAX / sizeof(struct bpe_pair_stats_node)) {
        return 0;
    }

    struct bpe_pair_stats_node *buf_nodes = bpe_malloc(stats_len * sizeof(struct bpe_pair_stats_node));

    size_t node_buf_i = 0;

    // Collect and count all adjacent pairs
    for (size_t i = 0; i < ctx->pieces_len; i++) {

        for (size_t j = 0; j < ctx->pieces[i].len - 1; j++) {
            buf_nodes[node_buf_i].pair.left = ctx->pieces[i].ids[j];
            buf_nodes[node_buf_i].pair.right = ctx->pieces[i].ids[j + 1];

            struct avl_node *_node = avl_insert(&tree, &buf_nodes[node_buf_i].node, pair_stat_cmp_func);
            if (_node != &buf_nodes[node_buf_i].node) {
                // Pair already in tree — increment existing counter
                struct bpe_pair_stats_node *_n = _get_entry(_node, struct bpe_pair_stats_node, node);
                _n->count++;
            }
            else {
                // New pair — initialize counter and advance the buffer index
                struct bpe_pair_stats_node *_n = _get_entry(_node, struct bpe_pair_stats_node, node);
                _n->count = 1;
                node_buf_i++;
            }
        }
    }

    if (tree.root) {
        // Find the node with the highest frequency
        struct bpe_pair_stats_node *p_max = buf_nodes;
        for (size_t i = 1; i < node_buf_i; i++) {
            struct bpe_pair_stats_node *p = buf_nodes + i;
            if (p->count > p_max->count) {
                p_max = p;
            }
        }

        pair->left = p_max->pair.left;
        pair->right = p_max->pair.right;
        unsigned long count = (unsigned long) p_max->count;

        bpe_free(buf_nodes);

        ctx->rank++;

        // Apply the winning merge to all pieces in-place
        merge_piece(ctx->pieces, ctx->pieces_len, pair, ctx->rank);
        return count;
    }

    bpe_free(buf_nodes);
    return 0;
}

/*
 * Pre-apply a sequence of merges to the training context.
 *
 * Used for "continue training" scenarios: load an existing model's merges,
 * apply them to the training data, then continue from that state.
 * Each merge increments the rank counter and transforms the pieces.
 */
void bpe_apply_merges(bpe_train_ctx_t *ctx, const bpe_pair_t *pairs, size_t pairs_len) {
    for (size_t i = 0; i < pairs_len; i++) {
        ctx->rank++;
        merge_piece(ctx->pieces, ctx->pieces_len, &pairs[i], ctx->rank);
    }
}

/*
 * Initialize one training piece from raw bytes.
 *
 * Each byte is initially its own token (IDs 0-255).
 * The resulting piece is a sequence of unsigned long token IDs.
 */
void bpe_train_ctx_idx_init(bpe_train_ctx_t *ctx, size_t idx, const char *bytes, size_t size) {
    ctx->pieces[idx].ids = bpe_malloc(size * sizeof(unsigned long));
    ctx->pieces[idx].len = size;
    for (size_t i = 0; i < size; i++) {
        ctx->pieces[idx].ids[i] = (unsigned long) ((unsigned char) bytes[i]);
    }
}

/*
 * Free all resources held by a training context.
 *
 * Frees each piece's ids[] array individually, then the caller
 * is responsible for freeing the pieces[] array itself.
 */
void bpe_train_ctx_free(bpe_train_ctx_t *ctx) {
    for (size_t i = 0; i < ctx->pieces_len; i++) {
        bpe_free(ctx->pieces[i].ids);
        ctx->pieces[i].ids = NULL;
    }
}
