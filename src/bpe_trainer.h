/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * BPE trainer — learns merge pairs from training data.
 *
 * Usage:
 *   1. Create a bpe_train_ctx_t and initialize each piece via
 *      bpe_train_ctx_idx_init() from raw bytes.
 *   2. Call bpe_get_max_count_pair() repeatedly to find and apply
 *      the most frequent pair — one training step per call.
 *      The function returns the pair, its rank, and its frequency,
 *      or 0 when no more merges are possible.
 *   3. Optionally use bpe_apply_merges() to pre-load existing merges
 *      for "continue training" scenarios.
 *   4. Free the context with bpe_train_ctx_free() when done.
 */

#ifndef SRC_BPE_TRAINER_H
#define SRC_BPE_TRAINER_H

#include "bpe_common.h"

/* Ranks start at 256 (0-255 are reserved for base byte tokens). */
#define BPE_TRAIN_ID_INIT 255

/*
 * Training context — holds all state during the BPE learning process.
 */
typedef struct {
    bpe_piece_t *pieces;     // array of training data chunks (owned by caller)
    size_t pieces_len;       // number of pieces

    unsigned long rank;      // next available token ID (increments each step)
} bpe_train_ctx_t;

/*
 * Initialize one training piece from raw bytes.
 *
 * Each byte becomes a base token ID (0-255). The piece's ids[] array
 * is allocated dynamically.
 *
 * Parameters:
 *   ctx   — training context
 *   idx   — piece index (0-indexed, must be < ctx->pieces_len)
 *   bytes — raw training data for this piece
 *   size  — number of bytes
 */
void bpe_train_ctx_idx_init(bpe_train_ctx_t *ctx, size_t idx, const char *bytes, size_t size);

/*
 * Free all resources within the training context.
 *
 * Frees each piece's ids[] array. The caller must free ctx->pieces
 * separately if it was dynamically allocated.
 */
void bpe_train_ctx_free(bpe_train_ctx_t *ctx);

/*
 * Perform one BPE training step.
 *
 * Finds the most frequent adjacent pair across all pieces,
 * applies the merge (replacing occurrences with a new token ID),
 * and returns the result.
 *
 * Returns:
 *   > 0  — the frequency (number of occurrences) of the merged pair
 *   = 0  — no more pairs available (training complete)
 *
 * On success, *pair is filled with the merged pair and ctx->rank
 * is incremented.
 */
unsigned long bpe_get_max_count_pair(bpe_pair_t *pair, bpe_train_ctx_t *ctx);

/*
 * Pre-apply an existing sequence of merges to the training context.
 *
 * Used for "continue training" from an imported model: apply the
 * existing merges first, then continue from that state.
 *
 * Parameters:
 *   ctx       — training context
 *   pairs     — array of merge pairs to apply in order
 *   pairs_len — number of pairs
 */
void bpe_apply_merges(bpe_train_ctx_t *ctx, const bpe_pair_t *pairs, size_t pairs_len);

#endif  /* SRC_BPE_TRAINER_H */
