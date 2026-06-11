/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * Common types and utilities shared across the BPE trainer and tokenizer.
 *
 * This module provides:
 *   - bpe_pair_t  — a BPE merge pair (left_id, right_id) → new_id
 *   - bpe_piece_t — a training chunk (sequence of token IDs)
 *   - bpe_check() — validates a merge pair sequence
 *   - bpe_malloc() / bpe_free() — memory management (wraps PyMem)
 *   - bpe_pair_cmp() — lexicographic pair comparison
 *   - bpe_utf8_length_from_head() — UTF-8 leading-byte decoder
 *
 * Design note:
 *   Only bpe_common.c and bpe_module.c are allowed to include <Python.h>.
 *   All other C files (trainer, tokenizer, tree) are pure standard C so
 *   they can be ported to embedded devices without modification.
 */

#ifndef SRC_BPE_COMMON_H
#define SRC_BPE_COMMON_H

#include "_tree_core.h"

/* --------------------------------------------------------------------------
 * BPE merge pair: (left_token_id, right_token_id)
 *
 * Each pair defines how two existing tokens merge into one new token.
 * The left and right IDs must reference tokens that already exist at
 * the time this merge is applied (IDs 0-255 for base bytes, then
 * sequential for each new token).  -------------------------------------------------------------------------- */
struct bpe_pair_s {
    unsigned long left;
    unsigned long right;
};

/* --------------------------------------------------------------------------
 * Training piece: a mutable sequence of token IDs.
 *
 * Represents one chunk of the training corpus (e.g. one regex-split
 * segment or one line of text).  During training, adjacent pairs are
 * merged in-place — the `len` shrinks and `ids` is compacted.
 * -------------------------------------------------------------------------- */
struct bpe_piece_s {
    unsigned long *ids;  /* dynamically allocated token ID sequence */
    size_t len;          /* current length (shrinks as merges apply)  */
};

typedef struct bpe_pair_s bpe_pair_t;
typedef struct bpe_piece_s bpe_piece_t;

/* --------------------------------------------------------------------------
 * Validate a BPE merge pair sequence.
 *
 * Checks two invariants:
 *   1. Each merge pair references only token IDs that already exist at
 *      its position in the sequence (IDs < 256 + index).
 *   2. No duplicate pairs exist in the sequence.
 *
 * Returns 1 if the merges are valid, 0 otherwise.
 * -------------------------------------------------------------------------- */
int bpe_check(const bpe_pair_t *pairs, size_t len);

/* --------------------------------------------------------------------------
 * UTF-8 leading-byte decoder.
 *
 * Given the leading byte of a potentially multi-byte UTF-8 sequence,
 * return the total number of bytes in that character.
 *
 *   0x00–0x7F → 1  (ASCII)
 *   0xC0–0xDF → 2  (2-byte sequence)
 *   0xE0–0xEF → 3  (3-byte sequence)
 *   0xF0–0xF4 → 4  (4-byte sequence, RFC 3629 limit)
 *   0x80–0xBF → 0  (continuation byte — not a start)
 *   0xF5–0xFF → 0  (invalid per RFC 3629)
 * -------------------------------------------------------------------------- */
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

    /* continuation bytes (0x80-0xBF) and invalid bytes (0xF5-0xFF) */
    return 0;
}

/* --------------------------------------------------------------------------
 * Lexicographic comparison of two merge pairs.
 *
 * Compares (left, right) lexicographically: left first, then right.
 * Returns -1, 0, or +1 (standard qsort / bsearch contract).
 * -------------------------------------------------------------------------- */
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
            return 0; /* equal */
        }
    }
}

/* --------------------------------------------------------------------------
 * Allocate memory using Python's PyMem_Malloc.
 *
 * Raises a Python MemoryError (via PyErr_NoMemory) on failure.
 * In the C extension context this integrates with Python's GC tracking.
 * -------------------------------------------------------------------------- */
void *bpe_malloc(size_t size);

/* --------------------------------------------------------------------------
 * Free memory allocated by bpe_malloc().
 * No-op if ptr is NULL.
 * -------------------------------------------------------------------------- */
void bpe_free(void *ptr);

#endif  /* SRC_BPE_COMMON_H */
