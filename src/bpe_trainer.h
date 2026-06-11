/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * BPE trainer — learns merge pairs from training data.
 *
 * ## Algorithm
 *
 * Starting from the 256 base byte tokens (IDs 0-255), each training step:
 *   1. Scans all adjacent pairs across all training pieces
 *   2. Builds an AVL tree counting pair frequencies
 *   3. Finds the most frequent pair
 *   4. Replaces all occurrences of that pair with a new token ID
 *   5. The new token ID becomes available for future merges
 *
 * ## Usage
 *
 *   1. Create a bpe_train_ctx_t and initialize each piece via
 *      bpe_train_ctx_idx_init().
 *   2. Call bpe_get_max_count_pair() repeatedly — one training step
 *      per call.  The function returns the pair, its rank, and its
 *      frequency, or 0 when no more merges are possible.
 *   3. Optionally use bpe_apply_merges() to pre-load existing merges
 *      for "continue training" scenarios.
 *   4. Free the context with bpe_train_ctx_free() when done.
 *
 * ## Pure C
 *
 * This module does NOT include <Python.h>.  It is portable to any
 * platform with a C99 compiler and the bpe_common / AVL tree modules.
 */

#ifndef SRC_BPE_TRAINER_H
#define SRC_BPE_TRAINER_H

#include "bpe_common.h"

/* Token IDs 0-255 are reserved for the base byte tokens.
 * The first learned merge gets ID 256. */
#define BPE_TRAIN_RANK_INIT 255

/* --------------------------------------------------------------------------
 * Training context — holds all state during BPE learning.
 * -------------------------------------------------------------------------- */
typedef struct {
    bpe_piece_t *pieces;      /* array of training data chunks (caller-owned) */
    size_t pieces_len;        /* number of pieces                              */

    unsigned long rank;       /* next available token ID (starts at 256)      */
} bpe_train_ctx_t;

/* --------------------------------------------------------------------------
 * Initialize one training piece from raw bytes.
 *
 * Each byte becomes a base token ID (0-255).  The piece's ids[] array
 * is allocated dynamically via bpe_malloc().
 *
 * Parameters:
 *   ctx   — training context
 *   idx   — piece index (0-indexed, must be < ctx->pieces_len)
 *   bytes — raw training data for this piece
 *   size  — number of bytes
 * -------------------------------------------------------------------------- */
void bpe_train_ctx_idx_init(bpe_train_ctx_t *ctx, size_t idx,
                            const char *bytes, size_t size);

/* --------------------------------------------------------------------------
 * Free all resources within the training context.
 *
 * Frees each piece's ids[] array.  The caller must free ctx->pieces
 * separately if it was dynamically allocated.
 * -------------------------------------------------------------------------- */
void bpe_train_ctx_free(bpe_train_ctx_t *ctx);

/* --------------------------------------------------------------------------
 * Perform one BPE training step.
 *
 * Finds the most frequent adjacent pair across all pieces, applies the
 * merge (replacing occurrences with a new token ID), and returns the
 * result.
 *
 * On success, *pair is filled with the winning pair and ctx->rank is
 * incremented.  Returns the frequency count (> 0), or 0 if no more
 * pairs are available (training complete).
 * -------------------------------------------------------------------------- */
unsigned long bpe_get_max_count_pair(bpe_pair_t *pair, bpe_train_ctx_t *ctx);

/* --------------------------------------------------------------------------
 * Pre-apply an existing sequence of merges to the training context.
 *
 * Used for "continue training" from an imported model.  Applies the
 * merges in order, transforming the pieces and advancing ctx->rank.
 *
 * Parameters:
 *   ctx       — training context
 *   pairs     — array of merge pairs to apply in order
 *   pairs_len — number of pairs
 * -------------------------------------------------------------------------- */
void bpe_apply_merges(bpe_train_ctx_t *ctx, const bpe_pair_t *pairs,
                      size_t pairs_len);

#endif  /* SRC_BPE_TRAINER_H */
