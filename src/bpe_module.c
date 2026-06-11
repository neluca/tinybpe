/*
 * Copyright (c) 2025-2026 Yinan Liao and other contributors.
 * SPDX-License-Identifier: MIT
 *
 * CPython extension module — Python bindings for the BPE C library.
 *
 * This is one of only two files (along with bpe_common.c) that include
 * <Python.h>.  It defines three Python types:
 *
 *   bpe.Trainer     — wraps bpe_train_ctx_t for BPE training
 *   bpe.Tokenizer   — wraps bpe_merges + bpe_vocab for encode/decode
 *   bpe.BytesRemap  — callable byte-level permutation for tiktoken compat
 *
 * All algorithmic work is delegated to the pure-C modules bpe_trainer
 * and bpe_tokenizer, which are portable to non-Python environments.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "bpe_trainer.h"
#include "bpe_tokenizer.h"

/* =========================================================================
 * Trainer
 * ========================================================================= */

typedef struct {
    PyObject_HEAD
    PyObject *list_merges;       /* Python list of (left, right) tuples */
    bpe_train_ctx_t ctx;         /* C training context                   */
} TrainerObject;

/* ---- Trainer.__init__(self, list_bytes) ---- */

static int trainer_init(TrainerObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"list_bytes", NULL};
    PyObject *list = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &list)) {
        return -1;
    }

    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError,
                        "Expected a list of bytes-like objects.");
        return -1;
    }

    Py_ssize_t list_len = PyList_Size(list);
    self->list_merges = NULL;

    if (list_len == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "The list must not be empty.");
        return -1;
    }

    self->ctx.rank = BPE_TRAIN_RANK_INIT;
    self->ctx.pieces_len = (size_t)list_len;
    self->ctx.pieces = bpe_malloc(list_len * sizeof(bpe_piece_t));

    for (Py_ssize_t i = 0; i < list_len; i++) {
        PyObject *item = PyList_GetItem(list, i);

        if (PyBytes_Check(item)) {
            Py_ssize_t size = PyBytes_Size(item);
            const char *bytes = PyBytes_AsString(item);
            bpe_train_ctx_idx_init(&self->ctx, i, bytes, (size_t)size);
        }
        else if (PyByteArray_Check(item)) {
            Py_ssize_t size = PyByteArray_Size(item);
            const char *bytes = PyByteArray_AsString(item);
            bpe_train_ctx_idx_init(&self->ctx, i, bytes, (size_t)size);
        }
        else {
            /* Only free pieces that were actually initialized (0..i-1) */
            for (Py_ssize_t j = 0; j < i; j++) {
                bpe_free(self->ctx.pieces[j].ids);
                self->ctx.pieces[j].ids = NULL;
            }
            bpe_free(self->ctx.pieces);
            self->ctx.pieces = NULL;
            self->ctx.pieces_len = 0;
            PyErr_SetString(PyExc_TypeError,
                            "Each element must be bytes or bytearray.");
            return -1;
        }
    }

    self->list_merges = PyList_New(0);
    return 0;
}

/* ---- Trainer.__dealloc__ ---- */

static void trainer_dealloc(TrainerObject *self) {
    if (self->ctx.pieces) {
        for (size_t i = 0; i < self->ctx.pieces_len; i++) {
            bpe_free(self->ctx.pieces[i].ids);
        }
        bpe_free(self->ctx.pieces);
        self->ctx.pieces = NULL;
    }

    Py_XDECREF(self->list_merges);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ---- Trainer.merges (getter) ---- */

static PyObject *trainer_get_merges(TrainerObject *self, void *Py_UNUSED(closure)) {
    Py_INCREF(self->list_merges);
    return self->list_merges;
}

/* ---- Trainer.n_merges (getter) ---- */

static PyObject *trainer_get_n_merges(TrainerObject *self, void *Py_UNUSED(closure)) {
    return PyLong_FromSsize_t(PyList_Size(self->list_merges));
}

/* ---- Trainer.step() → (pair, rank, freq) or None ---- */

static PyObject *trainer_step(TrainerObject *self, PyObject *Py_UNUSED(args)) {
    bpe_pair_t pair;
    unsigned long count = bpe_get_max_count_pair(&pair, &self->ctx);

    if (count) {
        PyObject *pair_tuple = Py_BuildValue("(ii)", pair.left, pair.right);
        if (PyList_Append(self->list_merges, pair_tuple) < 0) {
            Py_DECREF(pair_tuple);
            return NULL;
        }

        return Py_BuildValue("(Oii)", pair_tuple, self->ctx.rank, count);
    }

    Py_RETURN_NONE;
}

/* ---- Trainer.load_merges(merges) — for continue-training ---- */

static PyObject *trainer_load_merges(TrainerObject *self, PyObject *args,
                                     PyObject *kwds) {
    static char *kwlist[] = {"merges", NULL};

    /* Guards: must not already have merges */
    if (self->list_merges) {
        Py_ssize_t size = PyList_Size(self->list_merges);
        if (size != 0) {
            PyErr_SetString(PyExc_TypeError,
                            "Merges already loaded; cannot load again.");
            return NULL;
        }
    }

    PyObject *list_merges = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &list_merges)) {
        return NULL;
    }

    if (!PyList_Check(list_merges)) {
        PyErr_SetString(PyExc_TypeError,
                        "\"merges\" must be a list of (left, right) tuples.");
        return NULL;
    }

    Py_ssize_t merges_size = PyList_Size(list_merges);
    if (merges_size == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "The \"merges\" list must not be empty.");
        return NULL;
    }

    bpe_pair_t *pairs = bpe_malloc(merges_size * sizeof(bpe_pair_t));

    for (Py_ssize_t i = 0; i < merges_size; i++) {
        PyObject *item = PyList_GetItem(list_merges, i);
        if (!item || !PyTuple_Check(item) || PyTuple_Size(item) != 2) {
            bpe_free(pairs);
            PyErr_SetString(PyExc_TypeError,
                            "Each element must be a tuple of (left, right).");
            return NULL;
        }
        PyObject *left = PyTuple_GetItem(item, 0);
        PyObject *right = PyTuple_GetItem(item, 1);

        pairs[i].left = PyLong_AsUnsignedLong(left);
        pairs[i].right = PyLong_AsUnsignedLong(right);
        if (PyErr_Occurred()) {
            bpe_free(pairs);
            return NULL;
        }
    }

    if (!bpe_check(pairs, (size_t)merges_size)) {
        bpe_free(pairs);
        PyErr_SetString(PyExc_ValueError,
                        "Invalid merge sequence.");
        return NULL;
    }

    Py_XDECREF(self->list_merges);
    self->list_merges = list_merges;
    Py_INCREF(self->list_merges);

    bpe_apply_merges(&self->ctx, pairs, (size_t)merges_size);
    bpe_free(pairs);

    Py_RETURN_NONE;
}

/* =========================================================================
 * Tokenizer
 * ========================================================================= */

typedef struct {
    PyObject_HEAD
    PyObject *list_merges;              /* Python list of merge tuples      */
    PyObject *dict_special_tokens;      /* bytes → id  (or NULL)            */
    PyObject *dict_inverse_special;     /* id → bytes (or NULL)             */

    bpe_pair_t *pairs;                  /* C array of merge pairs           */
    size_t pairs_size;
    struct bpe_merges *merges;          /* AVL tree: pair → rank            */
    struct bpe_vocab *vocab;            /* flat array: id → bytes           */

    unsigned char bytes_cache[4];       /* streaming decode cache           */
    unsigned long bytes_cache_size;
} TokenizerObject;

/* ---- Tokenizer.__init__(self, merges, special_tokens=None) ---- */

static int tokenizer_init(TokenizerObject *self, PyObject *args,
                          PyObject *kwds) {
    static char *kwlist[] = {"merges", "special_tokens", NULL};
    PyObject *list_merges = NULL;
    PyObject *dict_special_tokens = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist,
                                     &list_merges, &dict_special_tokens)) {
        return -1;
    }

    if (!PyList_Check(list_merges)) {
        PyErr_SetString(PyExc_TypeError,
                        "\"merges\" must be a list of (left, right) tuples.");
        return -1;
    }

    Py_ssize_t merges_size = PyList_Size(list_merges);
    if (merges_size == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "The merges list must not be empty.");
        return -1;
    }

    /* Init all pointer fields to NULL for safe cleanup on error */
    self->pairs = NULL;
    self->merges = NULL;
    self->vocab = NULL;
    self->list_merges = NULL;
    self->dict_special_tokens = NULL;
    self->dict_inverse_special = NULL;

    /* ---- special tokens ---- */
    if (dict_special_tokens) {
        if (PyDict_Check(dict_special_tokens)
            && PyDict_Size(dict_special_tokens) != 0) {
            self->dict_special_tokens = dict_special_tokens;
            Py_INCREF(self->dict_special_tokens);

            /* Build inverse mapping: id → bytes */
            PyObject *inv = PyDict_New();
            PyObject *key, *value;
            Py_ssize_t pos = 0;
            while (PyDict_Next(dict_special_tokens, &pos, &key, &value)) {
                PyDict_SetItem(inv, value, key);
            }
            self->dict_inverse_special = inv;
        }
        else {
            self->dict_special_tokens = NULL;
            self->dict_inverse_special = NULL;
            PyErr_WarnEx(PyExc_UserWarning,
                         "special_tokens must be a non-empty dict.", 1);
        }
    }
    else {
        self->dict_special_tokens = NULL;
        self->dict_inverse_special = NULL;
    }

    /* ---- Copy pairs into C array, validating each element ---- */
    self->pairs_size = (size_t)merges_size;
    self->pairs = bpe_malloc(merges_size * sizeof(bpe_pair_t));

    for (Py_ssize_t i = 0; i < merges_size; i++) {
        PyObject *item = PyList_GetItem(list_merges, i);
        if (!item || !PyTuple_Check(item) || PyTuple_Size(item) != 2) {
            PyErr_SetString(PyExc_TypeError,
                            "Each element must be a tuple of (left, right).");
            return -1;
        }
        PyObject *left = PyTuple_GetItem(item, 0);
        PyObject *right = PyTuple_GetItem(item, 1);

        self->pairs[i].left = PyLong_AsUnsignedLong(left);
        self->pairs[i].right = PyLong_AsUnsignedLong(right);
        if (PyErr_Occurred()) {
            return -1;
        }
    }

    if (!bpe_check(self->pairs, self->pairs_size)) {
        bpe_free(self->pairs);
        self->pairs = NULL;
        PyErr_SetString(PyExc_ValueError, "Invalid merge sequence.");
        return -1;
    }

    self->list_merges = list_merges;
    Py_INCREF(self->list_merges);

    /* Build the merges search tree and vocab */
    self->merges = bpe_merges_build(self->pairs, self->pairs_size);
    if (self->merges == NULL) {
        return -1;
    }
    self->vocab = bpe_vocab_build(self->pairs, self->pairs_size);
    if (self->vocab == NULL) {
        bpe_merges_free(self->merges);
        self->merges = NULL;
        return -1;
    }
    self->bytes_cache_size = 0;

    return 0;
}

/* ---- Tokenizer.__dealloc__ ---- */

static void tokenizer_dealloc(TokenizerObject *self) {
    bpe_free(self->pairs);
    self->pairs = NULL;
    bpe_merges_free(self->merges);
    self->merges = NULL;
    bpe_vocab_free(self->vocab);
    self->vocab = NULL;

    Py_XDECREF(self->list_merges);
    Py_XDECREF(self->dict_special_tokens);
    Py_XDECREF(self->dict_inverse_special);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ---- Tokenizer.merges (getter) ---- */

static PyObject *tokenizer_get_merges(TokenizerObject *self,
                                      void *Py_UNUSED(closure)) {
    Py_INCREF(self->list_merges);
    return self->list_merges;
}

/* ---- Tokenizer.vocab (getter) → dict[int, bytes] ---- */

static PyObject *tokenizer_get_vocab(TokenizerObject *self,
                                     void *Py_UNUSED(closure)) {
    PyObject *vocab = PyDict_New();
    for (size_t i = 0; i < self->vocab->vocab_size; i++) {
        PyObject *key = PyLong_FromSize_t(i);
        PyObject *value = PyBytes_FromStringAndSize(
            (char *)self->vocab->tokens[i].bytes,
            (Py_ssize_t)self->vocab->tokens[i].size);
        PyDict_SetItem(vocab, key, value);
        Py_DECREF(key);
        Py_DECREF(value);
    }

    /* Merge in inverse special tokens (id → bytes) */
    if (self->dict_inverse_special) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(self->dict_inverse_special, &pos, &key, &value)) {
            PyDict_SetItem(vocab, key, value);
        }
    }

    return vocab;
}

/* ---- Tokenizer.n_vocab (getter) ---- */

static PyObject *tokenizer_get_n_vocab(TokenizerObject *self,
                                       void *Py_UNUSED(closure)) {
    Py_ssize_t special_size = 0;
    if (self->dict_special_tokens) {
        special_size = PyDict_Size(self->dict_special_tokens);
    }
    return PyLong_FromSize_t(self->vocab->vocab_size + (size_t)special_size);
}

/* ---- Tokenizer.encode(bytes) → list[int] ---- */

static PyObject *tokenizer_encode(TokenizerObject *self, PyObject *bytes_o) {
    /* Type check */
    if (!PyBytes_Check(bytes_o)) {
        PyErr_SetString(PyExc_TypeError, "encode() argument must be bytes.");
        return NULL;
    }

    /* Check for special token match */
    if (self->dict_special_tokens) {
        PyObject *token_id = PyDict_GetItem(self->dict_special_tokens, bytes_o);
        if (token_id) {
            Py_INCREF(token_id);
            PyObject *ids_list = PyList_New(1);
            PyList_SetItem(ids_list, 0, token_id);
            return ids_list;
        }
    }

    Py_ssize_t text_bytes_size = PyBytes_Size(bytes_o);
    if (text_bytes_size == 0) {
        return PyList_New(0);
    }
    char *text_bytes = PyBytes_AsString(bytes_o);

    size_t ids_len;
    unsigned long *ids = bpe_encode(&ids_len, self->merges,
                                    text_bytes, text_bytes_size);
    if (ids == NULL) {
        return NULL;
    }

    PyObject *ids_list = PyList_New((Py_ssize_t)ids_len);
    for (size_t i = 0; i < ids_len; i++) {
        PyObject *id = PyLong_FromUnsignedLong(ids[i]);
        PyList_SetItem(ids_list, (Py_ssize_t)i, id);
    }

    bpe_free(ids);
    return ids_list;
}

/* ---- Tokenizer.decode(list[int]) → bytes ---- */

static PyObject *tokenizer_decode(TokenizerObject *self, PyObject *list_ids) {
    Py_ssize_t size = PyList_Size(list_ids);
    if (size == 0) {
        return PyBytes_FromString("");
    }

    unsigned long *ids = bpe_malloc(size * sizeof(unsigned long));
    if (ids == NULL) {
        return NULL;
    }
    size_t ids_buf_len = 0;
    PyObject *result = PyBytes_FromString("");

    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item_id = PyList_GetItem(list_ids, i);
        unsigned long token_id = PyLong_AsUnsignedLong(item_id);

        if (token_id >= self->vocab->vocab_size) {
            /* Flush accumulated vocab tokens first */
            if (ids_buf_len) {
                size_t bytes_size;
                char *c_bytes = bpe_decode(&bytes_size, self->vocab,
                                           ids, ids_buf_len);
                if (c_bytes == NULL) {
                    bpe_free(ids);
                    Py_DECREF(result);
                    return NULL;
                }

                PyObject *chunk = PyBytes_FromStringAndSize(
                    c_bytes, (Py_ssize_t)bytes_size);
                PyBytes_Concat(&result, chunk);
                Py_DECREF(chunk);
                bpe_free(c_bytes);
                ids_buf_len = 0;
            }

            /* Look up special token */
            if (self->dict_inverse_special) {
                PyObject *special_bytes =
                    PyDict_GetItem(self->dict_inverse_special, item_id);
                if (special_bytes) {
                    PyBytes_Concat(&result, special_bytes);
                }
                else {
                    PyErr_WarnFormat(PyExc_UserWarning, 1,
                                     "Unknown token ID (%lu)", token_id);
                }
            }
            else {
                PyErr_WarnEx(PyExc_UserWarning, "No special_tokens defined.", 1);
            }
        }
        else {
            ids[ids_buf_len++] = token_id;
        }
    }

    /* Flush remaining vocab tokens */
    if (ids_buf_len) {
        size_t bytes_size;
        char *c_bytes = bpe_decode(&bytes_size, self->vocab,
                                   ids, ids_buf_len);
        if (c_bytes == NULL) {
            bpe_free(ids);
            Py_DECREF(result);
            return NULL;
        }

        PyObject *chunk = PyBytes_FromStringAndSize(
            c_bytes, (Py_ssize_t)bytes_size);
        PyBytes_Concat(&result, chunk);
        Py_DECREF(chunk);
        bpe_free(c_bytes);
    }

    bpe_free(ids);
    return result;
}

/* ---- Tokenizer.cache_decode(id) → bytes or None ---- */

static PyObject *tokenizer_cache_decode(TokenizerObject *self,
                                        PyObject *id_object) {
    /* Validate cache: flush invalid UTF-8 start bytes */
    if (self->bytes_cache_size
        && !bpe_utf8_length_from_head(self->bytes_cache[0])) {
        self->bytes_cache_size = 0;
    }

    unsigned long token_id = PyLong_AsUnsignedLong(id_object);

    if (token_id < self->vocab->vocab_size) {
        size_t bytes_size;
        char *c_bytes = bpe_decode_one(&bytes_size, self->vocab,
                                       token_id, self->bytes_cache,
                                       &self->bytes_cache_size);
        if (c_bytes == NULL) {
            return NULL;
        }

        PyObject *result;
        if (bytes_size) {
            result = PyBytes_FromStringAndSize(c_bytes,
                                               (Py_ssize_t)bytes_size);
        }
        else {
            result = Py_None;
            Py_INCREF(result);
        }

        bpe_free(c_bytes);
        return result;
    }
    else {
        /* Special token — flush any cached partial bytes first */
        if (self->dict_inverse_special) {
            PyObject *special_bytes =
                PyDict_GetItem(self->dict_inverse_special, id_object);
            if (special_bytes) {
                Py_INCREF(special_bytes);

                /* If cache has partial bytes, return them alone first.
                 * The special token bytes will be returned on the next call.
                 * However, in the current streaming API, a single token
                 * cannot produce two callbacks.  We flush cached bytes
                 * as incomplete fragments (risking garbled UTF-8), then
                 * return the special token.  This preserves the token
                 * boundary semantics at the cost of potential incomplete
                 * chars being lost when they span a special-token boundary. */
                if (self->bytes_cache_size) {
                    self->bytes_cache_size = 0;
                }
                return special_bytes;
            }
            PyErr_WarnFormat(PyExc_UserWarning, 1,
                             "Unknown token ID (%lu)", token_id);
        }
        else {
            PyErr_WarnEx(PyExc_UserWarning, "No special_tokens defined.", 1);
        }
    }

    Py_RETURN_NONE;
}

/* ---- Tokenizer.cache_clean() ---- */

static PyObject *tokenizer_cache_clean(TokenizerObject *self,
                                       PyObject *Py_UNUSED(args)) {
    self->bytes_cache_size = 0;
    Py_RETURN_NONE;
}

/* =========================================================================
 * BytesRemap — callable byte permutation (for tiktoken compat)
 * ========================================================================= */

typedef struct {
    PyObject_HEAD
    unsigned char _map[256];
} BytesRemapObject;

/* ---- BytesRemap.__init__(self, _remap) ---- */

static int bytes_remap_init(BytesRemapObject *self, PyObject *args,
                            PyObject *kwds) {
    static char *kwlist[] = {"_remap", NULL};
    PyObject *list = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &list)) {
        return -1;
    }

    if (!PyList_Check(list) || PyList_Size(list) != 256) {
        PyErr_SetString(PyExc_ValueError,
                        "\"_remap\" must be a list of exactly 256 integers.");
        return -1;
    }

    for (Py_ssize_t i = 0; i < 256; i++) {
        PyObject *item = PyList_GetItem(list, i);

        if (PyLong_Check(item)) {
            long val = PyLong_AsLong(item);
            if (val >= 0 && val < 256) {
                self->_map[i] = (unsigned char)val;
            }
            else {
                PyErr_SetString(PyExc_ValueError,
                                "All elements must be in range 0-255.");
                return -1;
            }
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "All elements must be integers.");
            return -1;
        }
    }

    return 0;
}

/* ---- BytesRemap.__dealloc__ ---- */

static void bytes_remap_dealloc(BytesRemapObject *self) {
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ---- BytesRemap.__call__(self, _bytes) → bytes ---- */

static PyObject *bytes_remap_call(BytesRemapObject *self, PyObject *args,
                                  PyObject *kwds) {
    static char *kwlist[] = {"_bytes", NULL};
    const char *bytes = NULL;
    Py_ssize_t bytes_size;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "y#", kwlist,
                                     &bytes, &bytes_size)) {
        return NULL;
    }

    unsigned char *buf = bpe_malloc((size_t)bytes_size);
    for (Py_ssize_t i = 0; i < bytes_size; i++) {
        buf[i] = self->_map[(unsigned char)bytes[i]];
    }

    PyObject *result = PyBytes_FromStringAndSize((const char *)buf,
                                                 bytes_size);
    bpe_free(buf);
    return result;
}

/* =========================================================================
 * Type definitions and getset/method tables
 * ========================================================================= */

static PyGetSetDef trainer_getset[] = {
    {"merges",   (getter)trainer_get_merges,    NULL,
     "List of learned merge pairs.", NULL},
    {"n_merges", (getter)trainer_get_n_merges,  NULL,
     "Number of merges learned so far.", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef trainer_methods[] = {
    {"step",        (PyCFunction)trainer_step,        METH_NOARGS,
     "Perform one BPE training step.\n\n"
     "Returns (pair, rank, frequency) or None if no more pairs."},
    {"load_merges", (PyCFunction)trainer_load_merges, METH_VARARGS | METH_KEYWORDS,
     "Load existing merges for continue-training."},
    {NULL}  /* Sentinel */
};

static PyTypeObject trainer_type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "bpe.Trainer",
    .tp_doc = PyDoc_STR("BPE trainer implemented in C.\n\n"
                         "Construct with a list of bytes or bytearray chunks."),
    .tp_basicsize = sizeof(TrainerObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)trainer_init,
    .tp_dealloc = (destructor)trainer_dealloc,
    .tp_getset = trainer_getset,
    .tp_methods = trainer_methods,
};

static PyGetSetDef tokenizer_getset[] = {
    {"merges",  (getter)tokenizer_get_merges,   NULL,
     "List of merge pairs defining the vocabulary.", NULL},
    {"vocab",   (getter)tokenizer_get_vocab,    NULL,
     "Vocabulary dict mapping token ID → bytes.", NULL},
    {"n_vocab", (getter)tokenizer_get_n_vocab,  NULL,
     "Total vocabulary size (256 + n_merges + n_special).", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef tokenizer_methods[] = {
    {"encode",       (PyCFunction)tokenizer_encode,       METH_O,
     "Encode bytes into a list of token IDs."},
    {"decode",       (PyCFunction)tokenizer_decode,       METH_O,
     "Decode a list of token IDs into bytes."},
    {"cache_decode", (PyCFunction)tokenizer_cache_decode, METH_O,
     "Streaming decode: accept one token ID, return decoded bytes or None."},
    {"cache_clean",  (PyCFunction)tokenizer_cache_clean,  METH_NOARGS,
     "Clear the streaming decode cache."},
    {NULL}  /* Sentinel */
};

static PyTypeObject tokenizer_type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "bpe.Tokenizer",
    .tp_doc = PyDoc_STR("BPE tokenizer implemented in C.\n\n"
                         "Construct with a list of merge pairs and optionally\n"
                         "a dict of special tokens (bytes → id)."),
    .tp_basicsize = sizeof(TokenizerObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)tokenizer_init,
    .tp_dealloc = (destructor)tokenizer_dealloc,
    .tp_getset = tokenizer_getset,
    .tp_methods = tokenizer_methods,
};

static PyTypeObject bytes_remap_type = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "bpe.BytesRemap",
    .tp_doc = PyDoc_STR("Byte-level permutation (0-255) for tiktoken compatibility.\n\n"
                         "Callable: remap(bytes) → bytes."),
    .tp_basicsize = sizeof(BytesRemapObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)bytes_remap_init,
    .tp_dealloc = (destructor)bytes_remap_dealloc,
    .tp_call = (ternaryfunc)bytes_remap_call,
};

/* =========================================================================
 * Module definition
 * ========================================================================= */

static PyModuleDef bpe_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "bpe",
    .m_doc = "TinyBPE C extension — ultra-fast BPE tokenizer and trainer.",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_bpe(void) {
    /* Ready the types */
    if (PyType_Ready(&trainer_type) < 0
        || PyType_Ready(&tokenizer_type) < 0
        || PyType_Ready(&bytes_remap_type) < 0) {
        return NULL;
    }

    /* Create module */
    PyObject *m = PyModule_Create(&bpe_module);
    if (m == NULL) {
        return NULL;
    }

    /* Add Trainer */
    Py_INCREF(&trainer_type);
    if (PyModule_AddObject(m, "Trainer", (PyObject *)&trainer_type) < 0) {
        Py_DECREF(&trainer_type);
        Py_DECREF(m);
        return NULL;
    }

    /* Add Tokenizer */
    Py_INCREF(&tokenizer_type);
    if (PyModule_AddObject(m, "Tokenizer", (PyObject *)&tokenizer_type) < 0) {
        Py_DECREF(&trainer_type);
        Py_DECREF(&tokenizer_type);
        Py_DECREF(m);
        return NULL;
    }

    /* Add BytesRemap */
    Py_INCREF(&bytes_remap_type);
    if (PyModule_AddObject(m, "BytesRemap", (PyObject *)&bytes_remap_type) < 0) {
        Py_DECREF(&trainer_type);
        Py_DECREF(&tokenizer_type);
        Py_DECREF(&bytes_remap_type);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
