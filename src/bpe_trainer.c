/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * BPE training algorithm implementation (pure C, no Python dependency).
 *
 * ## How It Works
 *
 * The trainer learns merge pairs from byte-level token sequences.
 * Starting from the 256 base byte tokens, each training step:
 *
 *   1. Scan all adjacent pairs across all training pieces
 *   2. Count frequencies using an AVL tree (O(n log n))
 *   3. Find the pair with the highest count (linear scan over unique pairs)
 *   4. Replace all occurrences of the winning pair with a new token ID
 *
 * ## Data Structures
 *
 *   bpe_train_ctx_t      — training context (pieces[], rank)
 *   bpe_pair_stats_node  — AVL tree node: {pair, count}
 *
 * ## Pure C Portability
 *
 * This file uses only standard C (C99) and the bpe_common / AVL tree
 * headers.  No Python.h dependency — portable to embedded devices.
 */

#include "bpe_trainer.h"

/* --------------------------------------------------------------------------
 * AVL tree node for pair frequency tracking.
 * -------------------------------------------------------------------------- */
struct bpe_pair_stats_node {
    struct avl_node node;
    bpe_pair_t pair;
    size_t count;
};

/* --------------------------------------------------------------------------
 * Replace all occurrences of a pair in-place across all training pieces.
 *
 * Scans each piece's ids[] array left-to-right, compacting as it goes.
 * When the pair is found, the two tokens are replaced by the merged ID
 * and the right half is skipped.  The piece length is updated to the
 * compacted size.
 *
 * Example: piece = [h, e, l, l, o], pair = (h, e), id = 256
 *          → piece = [256, l, l, o], len = 4
 * -------------------------------------------------------------------------- */
static inline void merge_piece(bpe_piece_t *pieces, size_t pieces_len,
                               const bpe_pair_t *pair, unsigned long id) {
    for (size_t i = 0; i < pieces_len; i++) {
        if (pieces[i].len > 1) {
            unsigned long *p_ids = pieces[i].ids;

            size_t new_ids_i = 0;
            for (size_t j = 0; j < pieces[i].len; j++) {
                if (p_ids[j] == pair->left
                    && j < pieces[i].len - 1
                    && p_ids[j + 1] == pair->right) {
                    p_ids[new_ids_i++] = id;
                    j++; /* skip the right half */
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

/* --------------------------------------------------------------------------
 * Perform one BPE training step — find and apply the most frequent pair.
 *
 * Algorithm:
 *   1. Count total adjacent pairs across all pieces (stats_len)
 *   2. Allocate buffer for all possible unique pair nodes
 *   3. Scan every piece: for each adjacent pair, insert into AVL tree;
 *      if already present, increment count; otherwise record as new
 *   4. Linear scan over unique pairs to find max frequency
 *   5. Apply the winning merge to all pieces
 *
 * Returns the frequency count (> 0) on success, or 0 if no pairs remain
 * (when stats_len == 0, i.e. no piece has len ≥ 2).
 * -------------------------------------------------------------------------- */
unsigned long bpe_get_max_count_pair(bpe_pair_t *pair, bpe_train_ctx_t *ctx) {
    struct avl_tree tree;
    avl_init(&tree);

    /* Count total adjacent pairs */
    size_t stats_len = 0;
    for (size_t i = 0; i < ctx->pieces_len; i++) {
        if (ctx->pieces[i].len >= 2) {
            stats_len += ctx->pieces[i].len - 1;
        }
    }

    if (stats_len == 0) {
        return 0; /* no more pairs available */
    }

    if (stats_len > SIZE_MAX / sizeof(struct bpe_pair_stats_node)) {
        return 0;
    }

    struct bpe_pair_stats_node *buf_nodes =
        bpe_malloc(stats_len * sizeof(struct bpe_pair_stats_node));

    size_t node_buf_i = 0;

    /* Collect and count all adjacent pairs */
    for (size_t i = 0; i < ctx->pieces_len; i++) {
        for (size_t j = 0; j + 1 < ctx->pieces[i].len; j++) {
            buf_nodes[node_buf_i].pair.left = ctx->pieces[i].ids[j];
            buf_nodes[node_buf_i].pair.right = ctx->pieces[i].ids[j + 1];

            struct avl_node *_node = avl_insert(
                &tree, &buf_nodes[node_buf_i].node, pair_stat_cmp_func);

            if (_node != &buf_nodes[node_buf_i].node) {
                /* Pair already in tree — increment existing counter */
                struct bpe_pair_stats_node *_n =
                    _get_entry(_node, struct bpe_pair_stats_node, node);
                _n->count++;
            }
            else {
                /* New pair — initialise counter and advance buffer index */
                struct bpe_pair_stats_node *_n =
                    _get_entry(_node, struct bpe_pair_stats_node, node);
                _n->count = 1;
                node_buf_i++;
            }
        }
    }

    if (tree.root) {
        /* Find the node with the highest frequency */
        struct bpe_pair_stats_node *p_max = buf_nodes;
        for (size_t i = 1; i < node_buf_i; i++) {
            struct bpe_pair_stats_node *p = buf_nodes + i;
            if (p->count > p_max->count) {
                p_max = p;
            }
        }

        pair->left = p_max->pair.left;
        pair->right = p_max->pair.right;
        unsigned long count = (unsigned long)p_max->count;

        bpe_free(buf_nodes);

        ctx->rank++;

        /* Apply the winning merge to all pieces in-place */
        merge_piece(ctx->pieces, ctx->pieces_len, pair, ctx->rank);
        return count;
    }

    bpe_free(buf_nodes);
    return 0;
}

/* --------------------------------------------------------------------------
 * Pre-apply a sequence of merges to the training context.
 *
 * Used for "continue training": load an existing model's merges, apply
 * them to the training data in order, then continue learning from that
 * state.  Each merge increments the rank and transforms the pieces.
 * -------------------------------------------------------------------------- */
void bpe_apply_merges(bpe_train_ctx_t *ctx, const bpe_pair_t *pairs,
                      size_t pairs_len) {
    for (size_t i = 0; i < pairs_len; i++) {
        ctx->rank++;
        merge_piece(ctx->pieces, ctx->pieces_len, &pairs[i], ctx->rank);
    }
}

/* --------------------------------------------------------------------------
 * Initialize one training piece from raw bytes.
 *
 * Each byte becomes a base token ID (0-255).  The resulting piece is
 * a sequence of unsigned long token IDs allocated via bpe_malloc().
 * -------------------------------------------------------------------------- */
void bpe_train_ctx_idx_init(bpe_train_ctx_t *ctx, size_t idx,
                            const char *bytes, size_t size) {
    ctx->pieces[idx].ids = bpe_malloc(size * sizeof(unsigned long));
    ctx->pieces[idx].len = size;
    for (size_t i = 0; i < size; i++) {
        ctx->pieces[idx].ids[i] = (unsigned long)((unsigned char)bytes[i]);
    }
}

/* --------------------------------------------------------------------------
 * Free all resources held by the training context.
 *
 * Frees each piece's ids[] array.  The caller is responsible for
 * freeing the pieces[] array itself (if dynamically allocated).
 * -------------------------------------------------------------------------- */
void bpe_train_ctx_free(bpe_train_ctx_t *ctx) {
    for (size_t i = 0; i < ctx->pieces_len; i++) {
        bpe_free(ctx->pieces[i].ids);
        ctx->pieces[i].ids = NULL;
    }
}
