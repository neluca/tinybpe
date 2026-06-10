/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * Common types and utilities shared between the BPE trainer and tokenizer.
 *
 * Provides:
 *   - bpe_pair_t / bpe_piece_t type definitions (in bpe_common.h)
 *   - bpe_check() — validates a merge pair sequence for correctness
 *   - bpe_malloc() / bpe_free() — memory management with Python integration
 */

#include <Python.h>
#include "bpe_common.h"

/* AVL tree node used by bpe_check() for duplicate detection. */
struct bpe_pair_node {
    struct avl_node node;
    bpe_pair_t pair;
};

static int pair_cmp_func(struct avl_node *a, struct avl_node *b) {
    struct bpe_pair_node *n1, *n2;

    n1 = _get_entry(a, struct bpe_pair_node, node);
    n2 = _get_entry(b, struct bpe_pair_node, node);

    return bpe_pair_cmp(&n1->pair, &n2->pair);
}

/*
 * Validate a BPE merge pair sequence.
 *
 * Checks two invariants:
 *   1. Each pair's left and right IDs must reference existing tokens
 *      (IDs 0-255 are base bytes; higher IDs are created by earlier merges).
 *   2. No duplicate merge pairs are present.
 *
 * Returns 1 if the merges are valid, 0 otherwise.
 */
int bpe_check(const bpe_pair_t *pairs, size_t len) {
    unsigned long max_id = 256;
    for (size_t i = 0; i < len; i++) {

        // The pair IDs at the current position must reference tokens
        // that already exist (i.e., are less than the current max_id).
        if (pairs[i].left >= max_id || pairs[i].right >= max_id) {
            return 0;
        }

        max_id++;
    }

    // Second pass: check for duplicate pairs using an AVL tree.
    struct bpe_pair_node *buf_nodes = bpe_malloc(len * sizeof(struct bpe_pair_node));
    struct avl_tree tree;
    avl_init(&tree);

    for (size_t i = 0; i < len; i++) {
        buf_nodes[i].pair.left = pairs[i].left;
        buf_nodes[i].pair.right = pairs[i].right;

        struct avl_node *node = avl_insert(&tree, &buf_nodes[i].node, pair_cmp_func);
        if (node != &buf_nodes[i].node) {
            bpe_free(buf_nodes);
            return 0; // Duplicate pairs are not allowed.
        }
    }

    bpe_free(buf_nodes);

    return 1;
}

/*
 * Allocate memory using Python's memory allocator (PyMem_Malloc).
 *
 * This ensures Python's garbage collector can track memory usage
 * and allows PyErr_NoMemory() to be raised on allocation failure.
 */
void *bpe_malloc(size_t size) {
    void *p = PyMem_Malloc(size); // fast allocator integrated with Python
    if (p == NULL) {
        PyErr_NoMemory(); // raises MemoryError in Python
    }
    return p;
}

/*
 * Free memory previously allocated by bpe_malloc().
 * No-op if p is NULL.
 */
void bpe_free(void *p) {
    if (p) {
        PyMem_Free(p);
    }
}
