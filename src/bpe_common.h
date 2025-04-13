/*
 * Copyright Yinan Liao. and other contributors. All rights reserved.
 */

#ifndef SRC_BPE_COMMON_H
#define SRC_BPE_COMMON_H

#include "_tree_core.h"

struct bpe_pair_s {
    unsigned long _1;
    unsigned long _2;
};

struct bpe_piece_s {
    unsigned long *ids;
    size_t len;
};

typedef struct bpe_pair_s bpe_pair_t;
typedef struct bpe_piece_s bpe_piece_t;

// Check whether the input merges are valid.
int bpe_check(const bpe_pair_t *pairs, size_t len);

// Check whether the input Token ID sequence is valid.
//int bpe_ids_check(const unsigned long *ids, size_t ids_size, size_t vocab_size);

static inline int bpe_utf8_head_check(unsigned char head_byte) {
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

    return 0;
}

// Default UTF-8
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

    // this "return" will not be executed.
    return 1; // fake ((head_byte & 0x80) == 0)
}

// A general implementation for comparing two pairs.
static inline int bpe_pair_cmp(const bpe_pair_t *p1, const bpe_pair_t *p2) {
    if (p1->_1 < p2->_1) {
        return -1;
    }
    else if (p1->_1 > p2->_1) {
        return 1;
    }
    else {
        if (p1->_2 < p2->_2) {
            return -1;
        }
        else if (p1->_2 > p2->_2) {
            return 1;
        }
        else {
            return 0; // equal
        }
    }
}

// A memory allocator with Python error handling
void *bpe_malloc(size_t size);

// Free memory
void bpe_free(void *p);

#endif  /* SRC_BPE_COMMON_H */
