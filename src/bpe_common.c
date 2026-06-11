/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * Common utilities: merge pair validation and Python-aware memory management.
 *
 * This is one of only two files (along with bpe_module.c) that include
 * <Python.h>.  The memory allocator uses PyMem_Malloc / PyMem_Free so
 * Python's garbage collector can account for C-level allocations.
 *
 * In an embedded / non-Python context, replace bpe_malloc / bpe_free
 * with wrappers around the platform's malloc / free — all other C files
 * (trainer, tokenizer, AVL tree) are pure standard C and need no changes.
 */

#include <Python.h>
#include "bpe_common.h"

/* --------------------------------------------------------------------------
 * AVL tree node for duplicate detection in bpe_check().
 * -------------------------------------------------------------------------- */
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

/* --------------------------------------------------------------------------
 * Validate a BPE merge pair sequence.
 *
 * Phase 1 — reachability check:
 *   Each pair's (left, right) IDs must be < (256 + index), i.e. must
 *   reference tokens that already exist at that point in the sequence.
 *
 * Phase 2 — uniqueness check:
 *   No duplicate pairs.  Uses an AVL tree with a single allocation for
 *   all nodes (O(n) memory, O(n log n) time).
 *
 * Returns 1 if valid, 0 otherwise.
 * -------------------------------------------------------------------------- */
int bpe_check(const bpe_pair_t *pairs, size_t len) {
    /* Phase 1: reachability */
    unsigned long max_id = 256;
    for (size_t i = 0; i < len; i++) {
        if (pairs[i].left >= max_id || pairs[i].right >= max_id) {
            return 0;
        }
        max_id++;
    }

    /* Phase 2: duplicate detection via AVL tree */
    struct bpe_pair_node *buf_nodes = bpe_malloc(len * sizeof(struct bpe_pair_node));
    struct avl_tree tree;
    avl_init(&tree);

    for (size_t i = 0; i < len; i++) {
        buf_nodes[i].pair.left = pairs[i].left;
        buf_nodes[i].pair.right = pairs[i].right;

        struct avl_node *node = avl_insert(&tree, &buf_nodes[i].node, pair_cmp_func);
        if (node != &buf_nodes[i].node) {
            /* duplicate found */
            bpe_free(buf_nodes);
            return 0;
        }
    }

    bpe_free(buf_nodes);
    return 1;
}

/* --------------------------------------------------------------------------
 * Allocate memory via PyMem_Malloc.
 *
 * PyMem_Malloc is a thin wrapper around the platform allocator that
 * Python's GC can track.  On failure, PyErr_NoMemory() sets a Python
 * MemoryError so the caller can simply return NULL.
 * -------------------------------------------------------------------------- */
void *bpe_malloc(size_t size) {
    void *p = PyMem_Malloc(size);
    if (p == NULL) {
        PyErr_NoMemory();
    }
    return p;
}

/* --------------------------------------------------------------------------
 * Free memory allocated by bpe_malloc().
 * No-op on NULL.
 * -------------------------------------------------------------------------- */
void bpe_free(void *ptr) {
    if (ptr) {
        PyMem_Free(ptr);
    }
}
