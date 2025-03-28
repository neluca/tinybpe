/*
 * Copyright Yinan Liao. and other contributors. All rights reserved.
 */

#include "bpe_tokenizer.h"
#include <string.h>

// merges avltree node
struct bpe_merges_node {
    struct avl_node node;
    bpe_pair_t pair;
    unsigned long id;
};

static int merges_cmp_func(struct avl_node *a, struct avl_node *b) {
    struct bpe_merges_node *n1, *n2;

    n1 = _get_entry(a, struct bpe_merges_node, node);
    n2 = _get_entry(b, struct bpe_merges_node, node);

    return bpe_pair_cmp(&n1->pair, &n2->pair);
}

struct bpe_merges *bpe_merges_build(bpe_pair_t *pairs, size_t len) {
    struct bpe_merges *merges = bpe_malloc(sizeof(struct bpe_merges));

    avl_init(&merges->tree); // init avltree

    struct bpe_merges_node *node_buf = bpe_malloc(len * sizeof(struct bpe_merges_node));
    merges->nodes_mem = node_buf;

    for (size_t i = 0; i < len; i++) {
        node_buf[i].pair = pairs[i];
        node_buf[i].id = (unsigned long) (256 + i);

        avl_insert(&merges->tree, &node_buf[i].node, merges_cmp_func); // insert node to avltree
    }

    return merges;
}

void bpe_merges_free(struct bpe_merges *p) {
    bpe_free(p->nodes_mem);
    bpe_free(p);
}

unsigned long *bpe_encode(size_t *ids_len, const struct bpe_merges *merges, const char *bytes, size_t bytes_size) {
    unsigned long *buf_ids = bpe_malloc(2 * bytes_size * sizeof(unsigned long));
    unsigned long *ctx_ids = buf_ids + bytes_size;

    // convert the bytes into a sequence of unsigned long .
    for (size_t i = 0; i < bytes_size; i++) {
        ctx_ids[i] = (unsigned long) ((unsigned char) bytes[i]);
    }

    size_t ids_size = bytes_size;
    size_t size = bytes_size;
    unsigned long *p = buf_ids;
    struct bpe_merges_node lookup;

    while (1) {
        for (size_t i = 0; i < size; i++) {
            *p = ctx_ids[i];

            if (i < size - 1) {
                lookup.pair._1 = ctx_ids[i];
                lookup.pair._2 = ctx_ids[i + 1];
                struct avl_node *node = avl_search((struct avl_tree *) &merges->tree,
                                                   (struct avl_node *) &lookup, merges_cmp_func);
                if (node != NULL) {
                    *p = ((struct bpe_merges_node *) node)->id;
                    ids_size--;
                    i++;
                }
            }
            p++;
        }
        if (ids_size == size) {
            break;
        }
        else {
            memcpy(ctx_ids, buf_ids, ids_size * sizeof(unsigned long));
            size = ids_size;
            p = buf_ids;
        }
    }

    *ids_len = ids_size;
    return buf_ids;
}

struct bpe_vocab *bpe_vocab_build(bpe_pair_t *pairs, size_t len) {
    struct bpe_vocab *vocab = bpe_malloc(sizeof(struct bpe_vocab));
    vocab->vocab_size = 256 + len;

    size_t total_bytes_size = 256;
    // The size of the bytes corresponding to each Token ID
    size_t *id_bytes_size = bpe_malloc(len * sizeof(size_t));

    // Calculate the total memory occupied by the vocabulary in bytes.
    for (size_t i = 0; i < len; i++) {
        size_t _size = 0;
        if (pairs[i]._1 < 256) {
            _size += 1;
        }
        else {
            _size += id_bytes_size[pairs[i]._1 - 256];
        }

        if (pairs[i]._2 < 256) {
            _size += 1;
        }
        else {
            _size += id_bytes_size[pairs[i]._2 - 256];
        }

        id_bytes_size[i] = _size;
        total_bytes_size += _size;
    }

    vocab->bytes_mem = bpe_malloc(total_bytes_size);
    vocab->tokens = bpe_malloc(vocab->vocab_size * sizeof(struct bpe_token_bytes));

    // mapping 0 - 255
    for (size_t i = 0; i < 256; i++) {
        vocab->bytes_mem[i] = (unsigned char) i;
        vocab->tokens[i].bytes = &vocab->bytes_mem[i];
        vocab->tokens[i].size = 1;
    }

    // mapping 255 - ...
    unsigned char *bytes_mem_p = vocab->bytes_mem + 256;
    for (size_t i = 0; i < len; i++) {

        memcpy(bytes_mem_p, vocab->tokens[pairs[i]._1].bytes, vocab->tokens[pairs[i]._1].size);
        size_t _size = vocab->tokens[pairs[i]._1].size;
        memcpy(bytes_mem_p + _size, vocab->tokens[pairs[i]._2].bytes, vocab->tokens[pairs[i]._2].size);

        vocab->tokens[i + 256].bytes = bytes_mem_p;
        vocab->tokens[i + 256].size = id_bytes_size[i];

        bytes_mem_p += id_bytes_size[i];
    }

    bpe_free(id_bytes_size);

    return vocab;
}

void bpe_vocab_free(struct bpe_vocab *p) {
    bpe_free(p->tokens);
    bpe_free(p->bytes_mem);
    bpe_free(p);
}

char *bpe_decode(size_t *bytes_size, const struct bpe_vocab *vocab, const unsigned long *ids, size_t ids_len) {
    size_t buf_size = 0; // Calculate the length of bytes.
    for (size_t i = 0; i < ids_len; i++) {
        buf_size += vocab->tokens[ids[i]].size;
    }

    *bytes_size = buf_size;
    char *buf_bytes = bpe_malloc(buf_size);
    char *p = buf_bytes;

    for (size_t i = 0; i < ids_len; i++) {
        memcpy(p, vocab->tokens[ids[i]].bytes, vocab->tokens[ids[i]].size);
        p += vocab->tokens[ids[i]].size;
    }

    return buf_bytes;
}

char *bpe_decode_one(size_t *bytes_size, const struct bpe_vocab *vocab,
                     unsigned long id, unsigned char *cache, unsigned long *cache_size) {
    size_t buf_size = vocab->tokens[id].size + (size_t) *cache_size;
    unsigned char *buf_bytes = bpe_malloc(buf_size);
    unsigned char *p = buf_bytes;
    if (*cache_size) {
        memcpy(p, cache, (size_t) *cache_size);
        p += (size_t) *cache_size;
    }

    memcpy(p, vocab->tokens[id].bytes, vocab->tokens[id].size);
    size_t utf8_size = 0;

    size_t i = bpe_utf8_length_from_head(buf_bytes[0]);
    while (i <= buf_size) {
        utf8_size = i;
        if (i == buf_size) {
            break;
        }

        i += bpe_utf8_length_from_head(buf_bytes[i]);
    }

    *bytes_size = utf8_size;
    *cache_size = (unsigned long) (buf_size - utf8_size);

    if (*cache_size) {
        memcpy(cache, buf_bytes + utf8_size, (size_t) *cache_size);
    }

    return (char *) buf_bytes;
}
