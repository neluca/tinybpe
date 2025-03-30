/*
 * Copyright Yinan Liao. and other contributors. All rights reserved.
 *
 * Reference: https://github.com/greensky00/avltree/blob/master/avltree/avltree.h
 */

#ifndef SRC_TREE_CORE_H
#define SRC_TREE_CORE_H

#include <stdint.h>
#include <stddef.h>

/* An AVL tree is a self-balancing binary search tree, in which the absolute value of
 * the height difference (balance factor) between the two subtrees of any node does not exceed 1.
 *
 * The space complexity is O(n)
 */

struct avl_node {
    struct avl_node *parent, *left, *right;
};

struct avl_tree {
    struct avl_node *root;
};

#define _get_entry(ELEM, STRUCT, MEMBER) \
        ((STRUCT *) ((uint8_t *) (ELEM) - offsetof(STRUCT, MEMBER)))

#define avl_parent(node) \
        ((struct avl_node *)((uintptr_t)(node)->parent & ~0x3))

#define avl_bf(node) (((int)((uintptr_t)(node)->parent & 0x3)) - 1)

// a < b : return -1
// a == b : return 0
// a > b : return 1
typedef int avl_cmp_func(struct avl_node *a, struct avl_node *b);

static inline void avl_init(struct avl_tree *tree) {
    tree->root = NULL;
}

// The operation of inserting a node into the tree
// has a complexity of O(log n)
struct avl_node *avl_insert(struct avl_tree *tree, struct avl_node *node, avl_cmp_func *func);

// The operation of searching for a node in the tree
// based on node information has a complexity of O(log n)
struct avl_node *avl_search(const struct avl_tree *tree, struct avl_node *node, avl_cmp_func *func);

#endif  /* SRC_TREE_CORE_H */
