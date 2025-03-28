/*
 * Copyright Yinan Liao. and other contributors. All rights reserved.
 */

#define APPLY_PYTHON
#ifdef APPLY_PYTHON
#include <Python.h>
#endif
#include "bpe_common.h"
#include <stdlib.h>

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

int bpe_check(const bpe_pair_t *pairs, size_t len) {
    unsigned long max_id = 256;
    for (size_t i = 0; i < len; i++) {

        // The pair ID at the current position cannot be greater than or equal
        // to the ID corresponding to the current position.
        if (pairs[i]._1 >= max_id || pairs[i]._2 >= max_id) {
            return 0;
        }

        max_id++;
    }

    struct bpe_pair_node *buf_nodes = bpe_malloc(len * sizeof(struct bpe_pair_node));
    struct avl_tree tree;
    avl_init(&tree);

    for (size_t i = 0; i < len; i++) {
        buf_nodes[i].pair._1 = pairs[i]._1;
        buf_nodes[i].pair._2 = pairs[i]._2;

        struct avl_node *node = avl_insert(&tree, &buf_nodes[i].node, pair_cmp_func);
        if (node != &buf_nodes[i].node) {
            bpe_free(buf_nodes);
            return 0; // Duplicate pairs cannot occur.
        }
    }

    bpe_free(buf_nodes);

    return 1;
}

//int bpe_ids_check(const unsigned long *ids, size_t ids_size, size_t vocab_size) {
//    unsigned long max_id = (unsigned long) vocab_size - 1;
//    for (size_t i = 0; i < ids_size; i++) {
//        // The Token ID sequence cannot be greater than
//        // the largest ID in the dictionary.
//        if (ids[i] > max_id) {
//            return 0;
//        }
//    }
//    return 1;
//}

void *bpe_malloc(size_t size) {
    void *p = malloc(size);
#ifdef APPLY_PYTHON
    if (p == NULL) {
        PyErr_NoMemory(); // Python error handling
    }
#endif
    return p;
}

void bpe_free(void *p) {
    if (p) {
        free(p);
    }
}
