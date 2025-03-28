/*
 * Copyright Yinan Liao. and other contributors. All rights reserved.
 *
 * Reference: https://github.com/greensky00/avltree/blob/master/avltree/avltree.c
 */

#include "_tree_core.h"

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define abs(a) ((a) < 0 ? (-(a)) : (a))

static inline void avl_set_parent(struct avl_node *node, struct avl_node *parent) {
    node->parent = (struct avl_node *) ((uintptr_t) parent | ((uintptr_t) node->parent & 0x3));
}

static inline void avl_set_bf(struct avl_node *node, int bf) {
    node->parent = (struct avl_node *) ((uintptr_t) avl_parent(node) | (uintptr_t) (bf + 1));
}

// MUST ensure that parent_bf <= 0
static inline struct avl_node *avl_rotate_LL(struct avl_node *parent, int parent_bf, int *child_bf, int *height_delta) {
    struct avl_node *child = parent->left;

    int child_left_bf = (child->left) ? (1) : (0);
    int child_right_bf = (child->right) ? (1) : (0);
    int parent_right_bf;

    if (*child_bf < 0) {
        // child->left > child->right
        child_left_bf = child_right_bf - (*child_bf);
        parent_right_bf = child_left_bf + 1 + parent_bf;

        if (height_delta) {
            *height_delta = max(child_left_bf, max(child_right_bf, parent_right_bf) + 1) - (child_left_bf + 1);
        }
    }
    else {
        // child->left <= child->right
        child_right_bf = child_left_bf + (*child_bf);
        parent_right_bf = child_right_bf + 1 + parent_bf;

        if (height_delta) {
            *height_delta = max(child_left_bf, max(child_right_bf, parent_right_bf) + 1) - (child_right_bf + 1);
        }
    }

    *child_bf = (max(child_right_bf, parent_right_bf) + 1) - child_left_bf;
    avl_set_bf(parent, parent_right_bf - child_right_bf);

    parent->left = child->right;
    if (child->right) {
        avl_set_parent(child->right, parent);
    }

    child->right = parent;
    avl_set_parent(child, avl_parent(parent));
    avl_set_parent(parent, child);

    return child;
}

// MUST ensure that parent_bf >= 0
static inline struct avl_node *avl_rotate_RR(struct avl_node *parent, int parent_bf, int *child_bf, int *height_delta) {
    struct avl_node *child = parent->right;

    int child_left_bf = (child->left) ? (1) : (0);
    int child_right_bf = (child->right) ? (1) : (0);
    int parent_left_bf;

    if (*child_bf < 0) {
        // child->left > child->right
        child_left_bf = child_right_bf - (*child_bf);
        parent_left_bf = child_left_bf + 1 - parent_bf;

        if (height_delta) {
            *height_delta = max(child_right_bf, max(child_left_bf, parent_left_bf) + 1) - (child_left_bf + 1);
        }
    }
    else {
        // child->left <= child->right
        child_right_bf = child_left_bf + (*child_bf);
        parent_left_bf = child_right_bf + 1 - parent_bf;

        if (height_delta) {
            *height_delta = max(child_right_bf, max(child_left_bf, parent_left_bf) + 1) - (child_right_bf + 1);
        }
    }

    *child_bf = child_right_bf - (max(child_left_bf, parent_left_bf) + 1);
    avl_set_bf(parent, child_left_bf - parent_left_bf);

    parent->right = child->left;
    if (child->left) {
        avl_set_parent(child->left, parent);
    }

    child->left = parent;
    avl_set_parent(child, avl_parent(parent));
    avl_set_parent(parent, child);

    return child;
}

static inline struct avl_node *avl_rotate_LR(struct avl_node *parent, int parent_bf) {
    int child_bf, height_delta = 0;
    struct avl_node *child = parent->left;
    struct avl_node *ret;

    if (child->right) {
        child_bf = avl_bf(child->right);
        parent->left = avl_rotate_RR(child, avl_bf(child), &child_bf, &height_delta);
    }
    else {
        child_bf = avl_bf(child);
    }

    ret = avl_rotate_LL(parent, parent_bf - height_delta, &child_bf, NULL);
    avl_set_bf(ret, child_bf);
    return ret;
}

static inline struct avl_node *avl_rotate_RL(struct avl_node *parent, int parent_bf) {
    int child_bf, height_delta = 0;
    struct avl_node *child = parent->right;
    struct avl_node *ret;

    if (child->left) {
        child_bf = avl_bf(child->left);
        parent->right = avl_rotate_LL(child, avl_bf(child), &child_bf, &height_delta);
    }
    else {
        child_bf = avl_bf(child);
    }

    ret = avl_rotate_RR(parent, parent_bf + height_delta, &child_bf, NULL);
    avl_set_bf(ret, child_bf);
    return ret;
}

#define _get_balance(node) ((node) ? avl_bf(node) : 0)

static struct avl_node *avl_balance_tree(struct avl_node *node, int bf) {
    int height_diff = _get_balance(node) + bf;
    int child_bf;

    if (node) {
        if (height_diff < -1 && node->left) {
            // balance left subtree
            if (_get_balance(node->left) <= 0) {
                child_bf = avl_bf(node->left);
                node = avl_rotate_LL(node, height_diff, &child_bf, NULL);
                avl_set_bf(node, child_bf);
            }
            else {
                node = avl_rotate_LR(node, height_diff);
            }
        }
        else if (height_diff > 1 && node->right) {
            // balance right subtree
            if (_get_balance(node->right) >= 0) {
                child_bf = avl_bf(node->right);
                node = avl_rotate_RR(node, height_diff, &child_bf, NULL);
                avl_set_bf(node, child_bf);
            }
            else {
                node = avl_rotate_RL(node, height_diff);
            }
        }
        else {
            avl_set_bf(node, avl_bf(node) + bf);
        }
    }

    return node;
}

struct avl_node *avl_insert(struct avl_tree *tree, struct avl_node *node, avl_cmp_func *func) {
    struct avl_node *node_original = node;
    struct avl_node *p = NULL;
    struct avl_node *cur = tree->root;

    while (cur) {
        int cmp = func(cur, node);
        p = cur;

        if (cmp > 0) {
            cur = cur->left;
        }
        else if (cmp < 0) {
            cur = cur->right;
        }
        else {
            // insert fail
            return cur;
        }
    }

    avl_set_parent(node, p);
    avl_set_bf(node, 0);
    node->left = node->right = NULL;

    if (p) {
        if (func(p, node) > 0) {
            p->left = node;
        }
        else {
            p->right = node;
        }
    }
    else {
        // no parent, make node as root
        tree->root = node;
    }

    // recursive balancing process, scan from leaf to root
    int bf = 0;
    int bf_old;
    while (node) {
        p = avl_parent(node);

        if (p) {
            // if parent exists
            bf_old = avl_bf(node);

            if (p->right == node) {
                node = avl_balance_tree(node, bf);
                p->right = node;
            }
            else {
                node = avl_balance_tree(node, bf);
                p->left = node;
            }

            // calculate balance facter BF for parent
            if (node->left == NULL && node->right == NULL) {
                // leaf node
                if (p->left == node) {
                    bf = -1;
                }
                else {
                    bf = 1;
                }
            }
            else {
                // index node
                bf = 0;
                int bf_node = avl_bf(node);
                if (abs(bf_old) < abs(bf_node)) {
                    // if ABS of balance factor increases
                    // cascade to parent
                    if (p->left == node) {
                        bf = -1;
                    }
                    else {
                        bf = 1;
                    }
                }
            }
        }
        else if (node == tree->root) {
            tree->root = avl_balance_tree(tree->root, bf);
            break;
        }
        if (bf == 0) {
            break;
        }
        node = p;
    }

    return node_original;
}

struct avl_node *avl_search(struct avl_tree *tree, struct avl_node *node, avl_cmp_func *func) {
    struct avl_node *p = tree->root;

    while (p) {
        int cmp = func(p, node);
        if (cmp > 0) {
            p = p->left;
        }
        else if (cmp < 0) {
            p = p->right;
        }
        else {
            return p;
        }
    }

    return NULL;
}
