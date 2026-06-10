/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * Generic AVL tree — a self-balancing binary search tree where the
 * absolute difference in height (balance factor) between the left
 * and right subtrees of any node never exceeds 1.
 *
 * The balance factor (-1, 0, +1) is packed into the low 2 bits of
 * the parent pointer, avoiding extra memory per node.
 *
 *   Space complexity: O(n)
 *   Insert: O(log n)
 *   Search: O(log n)
 *
 * Reference: https://github.com/greensky00/avltree/blob/master/avltree/avltree.h
 */

#ifndef SRC_TREE_CORE_H
#define SRC_TREE_CORE_H

#include <stdint.h>
#include <stddef.h>

/*
 * Compiler alignment notes:
 *   32-bit systems use 4-byte alignment
 *   64-bit systems use 8-byte alignment
 */

struct avl_node {
    struct avl_node *parent, *left, *right;
};

struct avl_tree {
    struct avl_node *root;
};

/* Retrieve the containing struct from an embedded avl_node pointer. */
#define _get_entry(NODE, STRUCT, ENTRY) \
        ((STRUCT *) (((uint8_t *) (NODE)) - offsetof(STRUCT, ENTRY)))

/* Extract the real parent pointer (low 2 bits are the balance factor). */
#define avl_parent(node) \
        ((struct avl_node *)((uintptr_t)(node)->parent & ~0x3))

/* Extract the balance factor: 0 → -1, 1 → 0, 2 → +1. */
#define avl_bf(node) (((int)((uintptr_t)(node)->parent & 0x3)) - 1)

/*
 * Comparison function type for AVL tree operations.
 *   a < b : return -1
 *   a == b: return 0
 *   a > b : return 1
 */
typedef int avl_cmp_func(struct avl_node *a, struct avl_node *b);

/* Initialize an empty AVL tree. */
static inline void avl_init(struct avl_tree *tree) {
    tree->root = NULL;
}

/*
 * Insert a node into the tree. Rebalances as needed.
 * Returns the inserted node, or the existing duplicate if found.
 * Complexity: O(log n)
 */
struct avl_node *avl_insert(struct avl_tree *tree, struct avl_node *node, avl_cmp_func *func);

/*
 * Search for a node in the tree using the comparison function.
 * Returns the matching node, or NULL if not found.
 * Complexity: O(log n)
 */
struct avl_node *avl_search(const struct avl_tree *tree, struct avl_node *node, avl_cmp_func *func);

#endif  /* SRC_TREE_CORE_H */
