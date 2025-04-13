/*
 * Copyright Yinan Liao. and other contributors. All rights reserved.
 */

#ifndef SRC_BPE_TRAINER_H
#define SRC_BPE_TRAINER_H

#include "bpe_common.h"

#define BPE_TRAIN_ID_INIT 255

// contextual information during the training process.
typedef struct {
    bpe_piece_t *pieces;
    size_t pieces_len;

    unsigned long rank;
} bpe_train_ctx_t;

// initialize the ctx with each piece one by one.
void bpe_train_ctx_idx_init(bpe_train_ctx_t *ctx, size_t idx, const char *bytes, size_t size);

// free the memory occupied by ctx.
void bpe_train_ctx_free(bpe_train_ctx_t *ctx);

// get the byte pair with the highest frequency of occurrence.
unsigned long bpe_get_max_count_pair(bpe_pair_t *pair, bpe_train_ctx_t *ctx);

// merge the pairs into the context information through external imports of merges.
void bpe_apply_merges(bpe_train_ctx_t *ctx, const bpe_pair_t *pairs, size_t pairs_len);

#endif  /* SRC_BPE_TRAINER_H */
