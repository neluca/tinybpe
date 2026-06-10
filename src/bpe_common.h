/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * Common types and utilities shared across the BPE trainer and tokenizer.
 */

#ifndef SRC_BPE_COMMON_H
#define SRC_BPE_COMMON_H

#include "_tree_core.h"

/*
 * A BPE merge pair: (left_token_id, right_token_id) → new_token_id.
 *
 * Each pair defines how two existing tokens merge into one.
 * The left and right IDs must reference tokens that already exist
 * at the time this merge was created.
 */
struct bpe_pair_s {
    unsigned long left;
    unsigned long right;
};

/*
 * A training piece: a sequence of token IDs representing one chunk
 * of the training corpus (e.g., one regex-split segment).
 */
struct bpe_piece_s {
    unsigned long *ids;  // dynamically allocated token ID sequence
    size_t len;          // current length (shrinks as merges are applied)
};

typedef struct bpe_pair_s bpe_pair_t;
typedef struct bpe_piece_s bpe_piece_t;

/*
 * Validate a merge pair sequence.
 *
 * Checks:
 *   1. Each pair references only existing token IDs
 *   2. No duplicate pairs exist
 *
 * Returns 1 if valid, 0 otherwise.
 */
int bpe_check(const bpe_pair_t *pairs, size_t len);

/*
 * Determine the number of bytes in a UTF-8 character from its leading byte.
 *
 *   0x00–0x7F → 1 byte  (ASCII)
 *   0xC0–0xDF → 2 bytes
 *   0xE0–0xEF → 3 bytes
 *   0xF0–0xF4 → 4 bytes
 *   0x80–0xBF → 0 (continuation byte — not a valid start)
 *   0xF5–0xFF → 0 (invalid per RFC 3629)
 */
static inline int bpe_utf8_length_from_head(unsigned char head_byte) {
    if ((head_byte & 0x80) == 0) {
        return 1;
    }
    else if ((head_byte & 0xE0) == 0xC0) {
        return 2;
    }
    else if ((head_byte & 0xF0) == 0xE0) {
        return 3;
    }
    else if ((head_byte & 0xF8) == 0xF0) {
        return 4;
    }

    // continuation bytes (0x80-0xBF) and invalid bytes (0xF5-0xFF)
    return 0;
}

/*
 * Lexicographic comparison of two merge pairs.
 *
 * Returns -1, 0, or +1 (standard comparison contract).
 */
static inline int bpe_pair_cmp(const bpe_pair_t *p1, const bpe_pair_t *p2) {
    if (p1->left < p2->left) {
        return -1;
    }
    else if (p1->left > p2->left) {
        return 1;
    }
    else {
        if (p1->right < p2->right) {
            return -1;
        }
        else if (p1->right > p2->right) {
            return 1;
        }
        else {
            return 0; // equal
        }
    }
}

/*
 * Allocate memory using Python's PyMem_Malloc.
 * Sets a Python MemoryError on failure (via PyErr_NoMemory).
 */
void *bpe_malloc(size_t size);

/*
 * Free memory allocated by bpe_malloc().
 * No-op if p is NULL.
 */
void bpe_free(void *p);

#endif  /* SRC_BPE_COMMON_H */
