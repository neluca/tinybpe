/*
 * Copyright Yinan Liao. and other contributors. All rights reserved.
 */

#include "bpe_trainer.h"

// statistics node
struct bpe_pair_stat_node {
    struct avl_node node;
    bpe_pair_t pair;
    size_t rank;
};

// replace the pair in the piece with the corresponding id.
static inline void merge_piece(bpe_piece_t *pieces, size_t pieces_len, const bpe_pair_t *pair, unsigned long id) {
    for (size_t i = 0; i < pieces_len; i++) {
        if (pieces[i].len > 1) {
            unsigned long *old_ids = pieces[i].ids;
            unsigned long *new_ids = pieces[i].ids;

            size_t new_ids_i = 0;
            for (size_t j = 0; j < pieces[i].len; j++) {
                if (old_ids[j] == pair->_1 && j < pieces[i].len - 1 && old_ids[j + 1] == pair->_2) {
                    new_ids[new_ids_i++] = id;
                    j++;
                }
                else {
                    new_ids[new_ids_i++] = old_ids[j];
                }
            }
            pieces[i].ids = new_ids;
            pieces[i].len = new_ids_i;
        }
    }
}

static int pair_stat_cmp_func(struct avl_node *a, struct avl_node *b) {
    struct bpe_pair_stat_node *n1, *n2;

    n1 = _get_entry(a, struct bpe_pair_stat_node, node);
    n2 = _get_entry(b, struct bpe_pair_stat_node, node);

    return bpe_pair_cmp(&n1->pair, &n2->pair);
}


unsigned long bpe_get_max_rank_pair(bpe_pair_t *pair, bpe_train_ctx_t *ctx) {
    struct avl_tree tree;
    avl_init(&tree);

    size_t stat_len = 0;
    for (size_t i = 0; i < ctx->pieces_len; i++) {
        stat_len += ctx->pieces[i].len - 1;
    }

    struct bpe_pair_stat_node *buf_nodes = bpe_malloc(stat_len * sizeof(struct bpe_pair_stat_node));

    size_t node_buf_i = 0;

    // statistics
    for (size_t i = 0; i < ctx->pieces_len; i++) {

        for (size_t j = 0; j < ctx->pieces[i].len - 1; j++) {
            buf_nodes[node_buf_i].pair._1 = ctx->pieces[i].ids[j];
            buf_nodes[node_buf_i].pair._2 = ctx->pieces[i].ids[j + 1];

            struct avl_node *node = avl_insert(&tree, &buf_nodes[node_buf_i].node, pair_stat_cmp_func);
            if (node != &buf_nodes[node_buf_i].node) {
                ((struct bpe_pair_stat_node *) node)->rank++; // in
            }
            else {
                ((struct bpe_pair_stat_node *) node)->rank = 1; // not in
                node_buf_i++;
            }
        }
    }

    if (tree.root) {
        // find the node with the highest frequency of occurrence.
        struct bpe_pair_stat_node *p_max = buf_nodes;
        for (size_t i = 1; i < node_buf_i; i++) {
            struct bpe_pair_stat_node *p = buf_nodes + i;
            if (p->rank > p_max->rank) {
                p_max = p;
            }
        }

        pair->_1 = p_max->pair._1;
        pair->_2 = p_max->pair._2;
        unsigned long rank = (unsigned long) p_max->rank;

        bpe_free(buf_nodes);

        ctx->id++;

        merge_piece(ctx->pieces, ctx->pieces_len, pair, ctx->id);
        return rank;
    }

    bpe_free(buf_nodes);
    return 0;
}

void bpe_apply_merges(bpe_train_ctx_t *ctx, const bpe_pair_t *pairs, size_t pairs_len) {
    for (size_t i = 0; i < pairs_len; i++) {
        ctx->id++;
        merge_piece(ctx->pieces, ctx->pieces_len, &pairs[i], ctx->id);
    }
}

void bpe_train_ctx_idx_init(bpe_train_ctx_t *ctx, size_t idx, const char *bytes, size_t size) {
    ctx->pieces[idx].ids = bpe_malloc(size * sizeof(unsigned long));
    ctx->pieces[idx].len = size;
    for (size_t i = 0; i < size; i++) {
        ctx->pieces[idx].ids[i] = (unsigned long) ((unsigned char) bytes[i]); //
    }
}

void bpe_train_ctx_free(bpe_train_ctx_t *ctx) {
    for (size_t i = 0; i < ctx->pieces_len; i++) {
        bpe_free(ctx->pieces[i].ids);
    }
}
