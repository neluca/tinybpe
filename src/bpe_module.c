/*
 * Copyright Yinan Liao. and other contributors. All rights reserved.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "bpe_trainer.h"
#include "bpe_tokenizer.h"

typedef struct {
    PyObject_HEAD
    PyObject *list_merges;

    bpe_train_ctx_t ctx;
} TrainerObject;

typedef struct {
    PyObject_HEAD
    PyObject *list_merges;
    PyObject *dict_special_tokens;
    PyObject *dict_inverse_special_tokens;

    bpe_pair_t *pairs;
    size_t pairs_size;
    struct bpe_merges *merges;
    struct bpe_vocab *vocab;

    unsigned char bytes_cache[4];
    unsigned long bytes_cache_size;
} TokenizerObject;

static int Trainer_init(TrainerObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"list_bytes", NULL};
    PyObject *list = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &list)) {
        return -1;
    }

    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "The input argument must be a list containing bytes-like objects.");
        return -1;
    }

    self->list_merges = PyList_New(0); // yes incref
    Py_ssize_t list_len = PyList_Size(list);

    if (list_len == 0) {
        PyErr_SetString(PyExc_Exception,
                        "The list must not be empty, and the objects in the list must be of bytes-like type.");
        return -1;
    }

    self->ctx.id = BPE_TRAIN_ID_INIT;
    self->ctx.pieces_len = (size_t) list_len;
    self->ctx.pieces = bpe_malloc(list_len * sizeof(bpe_piece_t));

    for (Py_ssize_t i = 0; i < list_len; i++) {
        PyObject *item = PyList_GetItem(list, i);

        if (PyBytes_Check(item)) {
            Py_ssize_t size = PyBytes_Size(item);
            const char *bytes = PyBytes_AsString(item);
            bpe_train_ctx_idx_init(&self->ctx, i, bytes, (size_t) size);
        }

        else if (PyByteArray_Check(item)) {
            Py_ssize_t size = PyByteArray_Size(item);
            const char *bytes = PyByteArray_AsString(item);
            bpe_train_ctx_idx_init(&self->ctx, i, bytes, (size_t) size);
        }

        else {
            PyErr_SetString(PyExc_TypeError, "The objects in the list must be of bytes-like type.");

            bpe_train_ctx_free(&self->ctx);
            bpe_free(self->ctx.pieces);
            return -1;
        }
    }

    return 0;
}

static void Trainer_dealloc(TrainerObject *self) {
    bpe_train_ctx_free(&self->ctx);
    bpe_free(self->ctx.pieces);

    Py_DECREF(self->list_merges);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *Trainer_Get_merges(TrainerObject *self, void *Py_UNUSED(closure)) {
    return Py_NewRef(self->list_merges);  // yes incref
}

static PyObject *Trainer_Get_merges_size(TrainerObject *self, void *Py_UNUSED(closure)) {
    Py_ssize_t size = PyList_Size(self->list_merges);
    return PyLong_FromSsize_t(size); // yes incref
}

static PyObject *Trainer_step(TrainerObject *self, PyObject *Py_UNUSED(args)) {
    bpe_pair_t pair;
    unsigned long rank = bpe_get_max_rank_pair(&pair, &self->ctx);

    if (rank) {
        PyObject *pair_tuple = Py_BuildValue("(ii)", pair._1, pair._2); // yes incref
        PyList_Append(self->list_merges, pair_tuple); // yes incref

        return Py_BuildValue("(Oii)", pair_tuple, self->ctx.id, rank); // yes incref
    }

    return Py_None;
}

static PyObject *Trainer_load_merges(TrainerObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"merges", NULL};

    if (self->list_merges) {
        Py_ssize_t size = PyList_Size(self->list_merges);
        if (size != 0) {
            PyErr_SetString(PyExc_TypeError, "The \"merges\" already exist.");
            return NULL;
        }
    }

    PyObject *list_merges = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &list_merges)) {
        return NULL;
    }

    if (!PyList_Check(list_merges)) {
        PyErr_SetString(PyExc_TypeError, "The \"merges\" must be a pairs list.");
        return NULL;
    }

    Py_ssize_t size = PyList_Size(list_merges);
    bpe_pair_t *pairs = bpe_malloc(size * sizeof(bpe_pair_t));

    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PyList_GetItem(list_merges, i); // no incref
        PyObject *item_1 = PyTuple_GetItem(item, 0); // no incref
        PyObject *item_2 = PyTuple_GetItem(item, 1); // no incref

        pairs[i]._1 = PyLong_AsUnsignedLong(item_1);
        pairs[i]._2 = PyLong_AsUnsignedLong(item_2);

        if ((int) pairs[i]._1 < 0 || (int) pairs[i]._2 < 0) {

            PyErr_SetString(PyExc_ValueError, "The \"merges\" must be positive integer.");
            return NULL;
        }
    }

    if (!bpe_check(pairs, (size_t) size)) {
        bpe_free(pairs);
        PyErr_SetString(PyExc_ValueError, "The provided merges are not valid.");
        return Py_None;
    }

    Py_DECREF(self->list_merges);
    self->list_merges = list_merges;
    Py_INCREF(self->list_merges);

    bpe_apply_merges(&self->ctx, pairs, (size_t) size);
    bpe_free(pairs);

    return Py_None;
}

static int Tokenizer_init(TokenizerObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"merges", "special_tokens", NULL};
    PyObject *list_merges = NULL;
    PyObject *dict_special_tokens = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &list_merges, &dict_special_tokens)) {
        return -1;
    }

    if (!PyList_Check(list_merges)) {
        PyErr_SetString(PyExc_TypeError, "The \"merges\" must be a pairs list.");
        return -1;
    }

    if (dict_special_tokens) {
        if (PyDict_Check(dict_special_tokens)) {
            self->dict_special_tokens = dict_special_tokens;
            Py_INCREF(self->dict_special_tokens);

            PyObject *dict_inverse_special_tokens = PyDict_New(); // yes incref

            PyObject *key, *value;
            Py_ssize_t pos = 0;
            while (PyDict_Next(dict_special_tokens, &pos, &key, &value)) { // no incref
                PyDict_SetItem(dict_inverse_special_tokens, value, key); // yes incref
            }

            self->dict_inverse_special_tokens = dict_inverse_special_tokens;
        }
        else {
            self->dict_special_tokens = NULL;
            self->dict_inverse_special_tokens = NULL;
            PyErr_WarnEx(PyExc_UserWarning, "special_tokens must be a dict.", 1);
        }
    }
    else {
        self->dict_special_tokens = NULL;
        self->dict_inverse_special_tokens = NULL;
    }

    Py_ssize_t size = PyList_Size(list_merges);

    self->pairs_size = (size_t) size;
    self->pairs = bpe_malloc(size * sizeof(bpe_pair_t));

    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item = PyList_GetItem(list_merges, i); // no incref
        PyObject *item_1 = PyTuple_GetItem(item, 0); // no incref
        PyObject *item_2 = PyTuple_GetItem(item, 1); // no incref

        self->pairs[i]._1 = PyLong_AsUnsignedLong(item_1);
        self->pairs[i]._2 = PyLong_AsUnsignedLong(item_2);

        if ((int) self->pairs[i]._1 < 0 || (int) self->pairs[i]._2 < 0) {

            PyErr_SetString(PyExc_ValueError, "The \"merges\" must be positive integer.");
            return -1;
        }
    }

    if (!bpe_check(self->pairs, self->pairs_size)) {
        bpe_free(self->pairs); // Release in advance to avoid memory leaks.
        PyErr_SetString(PyExc_ValueError, "The provided merges are not valid.");
        return -1;
    }

    self->list_merges = list_merges;
    Py_INCREF(self->list_merges);

    self->merges = bpe_merges_build(self->pairs, self->pairs_size);
    self->vocab = bpe_vocab_build(self->pairs, self->pairs_size);
    self->bytes_cache_size = 0;

    return 0;
}

static void Tokenizer_dealloc(TokenizerObject *self) {
    bpe_free(self->pairs);
    bpe_merges_free(self->merges);
    bpe_vocab_free(self->vocab);

    Py_DECREF(self->list_merges);
    Py_XDECREF(self->dict_special_tokens);
    Py_XDECREF(self->dict_inverse_special_tokens);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *Tokenizer_Get_merges(TokenizerObject *self, void *Py_UNUSED(closure)) {
    return Py_NewRef(self->list_merges); // yes incref
}

static PyObject *Tokenizer_Get_vocab(TokenizerObject *self, void *Py_UNUSED(closure)) {
    PyObject *vocab = PyDict_New();
    for (size_t i = 0; i < self->vocab->vocab_size; i++) {
        PyObject *key = PyLong_FromSize_t(i); // yes incref
        PyObject *value = PyBytes_FromStringAndSize((char *) self->vocab->tokens[i].bytes,     // yes incref
                                                    (Py_ssize_t) self->vocab->tokens[i].size);
        PyDict_SetItem(vocab, key, value); // yes incref
        Py_DECREF(key);
        Py_DECREF(value);
    }

    if (self->dict_inverse_special_tokens) {

        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(self->dict_inverse_special_tokens, &pos, &key, &value)) { // no incref
            PyDict_SetItem(vocab, key, value); // yes incref
        }
    }

    return vocab;
}

static PyObject *Tokenizer_Get_size(TokenizerObject *self, void *Py_UNUSED(closure)) {
    Py_ssize_t dict_special_tokens_size = 0;
    if (self->dict_special_tokens) {
        dict_special_tokens_size = PyDict_Size(self->dict_special_tokens);
    }

    size_t size = self->vocab->vocab_size + (size_t) dict_special_tokens_size;
    return PyLong_FromSize_t(size);
}

static PyObject *Tokenizer_encode(TokenizerObject *self, PyObject *bytes_o) {
    if (self->dict_special_tokens) {
        PyObject *token_id = PyDict_GetItem(self->dict_special_tokens, bytes_o); // no incref

        if (token_id) {
            Py_INCREF(token_id);
            PyObject *ids_list = PyList_New(1); // yes incref
            PyList_SetItem(ids_list, 0, token_id);
            return ids_list;
        }
    }

    Py_ssize_t text_bytes_size = PyBytes_Size(bytes_o);
    if (text_bytes_size == 0) {
        return PyList_New(0); // yes incref
    }
    char *text_bytes = PyBytes_AsString(bytes_o);

    size_t ids_len;
    unsigned long *ids = bpe_encode(&ids_len, self->merges, text_bytes, text_bytes_size);

    PyObject *ids_list = PyList_New((Py_ssize_t) ids_len);
    for (size_t i = 0; i < ids_len; i++) {
        PyObject *id = PyLong_FromUnsignedLong(ids[i]); // yes incref
        PyList_SetItem(ids_list, (Py_ssize_t) i, id); // no incref
    }

    bpe_free(ids);

    return ids_list;
}

static PyObject *Tokenizer_decode(TokenizerObject *self, PyObject *list_ids) {
    Py_ssize_t size = PyList_Size(list_ids);
    if (size == 0) {
        return PyBytes_FromString(""); // yes incref
    }

    unsigned long *ids = bpe_malloc(size * sizeof(unsigned long));
    size_t ids_size = 0;
    PyObject *bytes = PyBytes_FromString(""); // yes incref

    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item_id = PyList_GetItem(list_ids, i);
        unsigned long token_id = PyLong_AsLong(item_id);

        if (token_id >= self->vocab->vocab_size) {

            if (ids_size) {
                size_t bytes_size;
                char *c_bytes = bpe_decode(&bytes_size, self->vocab, ids, ids_size);

                PyBytes_Concat(&bytes, PyBytes_FromStringAndSize(c_bytes, (Py_ssize_t) bytes_size)); // no incref

                bpe_free(c_bytes);
                ids_size = 0;
            }

            if (self->dict_inverse_special_tokens) {
                PyObject *special_bytes = PyDict_GetItem(self->dict_inverse_special_tokens, item_id); // no incref

                if (special_bytes) {
                    Py_INCREF(special_bytes);

                    PyBytes_Concat(&bytes, special_bytes); // no incref
                }
                else {
                    PyErr_WarnFormat(PyExc_UserWarning, 1, "Unknown Token ID %lu \n", token_id);
                }
            }
            else {
                PyErr_WarnEx(PyExc_UserWarning, "No special_tokens.", 1);
            }
        }
        else {
            ids[ids_size++] = token_id;
        }
    }

    if (ids_size) {
        size_t bytes_size;
        char *c_bytes = bpe_decode(&bytes_size, self->vocab, ids, ids_size);

        PyBytes_Concat(&bytes, PyBytes_FromStringAndSize(c_bytes, (Py_ssize_t) bytes_size)); // no incref

        bpe_free(c_bytes);
    }

    bpe_free(ids);

    return bytes;
}

static PyObject *Tokenizer_cache_decode(TokenizerObject *self, PyObject *id_object) {
    if (self->bytes_cache_size && !bpe_utf8_head_check(self->bytes_cache[0])) {
        self->bytes_cache_size = 0;
    }
    unsigned long token_id = PyLong_AsLong(id_object);

    if (token_id < self->vocab->vocab_size) {
        size_t bytes_size;
        char *c_bytes = bpe_decode_one(&bytes_size, self->vocab, token_id, self->bytes_cache, &self->bytes_cache_size);
        PyObject *bytes = Py_None;

        if (bytes_size) {
            bytes = PyBytes_FromStringAndSize(c_bytes, (Py_ssize_t) bytes_size); // yes incref
        }

        bpe_free(c_bytes);

        return bytes;
    }
    else {
        if (self->dict_inverse_special_tokens) {
            PyObject *special_bytes = PyDict_GetItem(self->dict_inverse_special_tokens, id_object); // no incref
            if (special_bytes) {
                Py_INCREF(special_bytes);
                self->bytes_cache_size = 0;
                return special_bytes;
            }
            else {
                PyErr_WarnFormat(PyExc_UserWarning, 1, "Unknown Token ID %lu \n", token_id);
            }
        }
        else {
            PyErr_WarnEx(PyExc_UserWarning, "No special_tokens.", 1);
        }
    }

    Py_RETURN_NONE;
}

static PyObject *Tokenizer_cache_clean(TokenizerObject *self, PyObject *Py_UNUSED(args)) {
    self->bytes_cache_size = 0;
    Py_RETURN_NONE;
}

static PyGetSetDef Trainer_getset[] = {
        {"merges",      (getter) Trainer_Get_merges,      NULL, "The merges list of the BPE algorithm", NULL},
        {"merges_size", (getter) Trainer_Get_merges_size, NULL, "The length of the merges",             NULL},
        {NULL}  /* Sentinel */
};

static PyGetSetDef Tokenizer_getset[] = {
        {"merges", (getter) Tokenizer_Get_merges, NULL, "The merges list of the BPE algorithm", NULL},
        {"vocab",  (getter) Tokenizer_Get_vocab,  NULL, "Vocabulary dict of the BPE algorithm", NULL},
        {"size",   (getter) Tokenizer_Get_size,   NULL, "The length of the vocabulary",         NULL},
        {NULL}  /* Sentinel */
};

static PyMethodDef Trainer_methods[] = {
        {"step",        (PyCFunction) Trainer_step,        METH_NOARGS, "During training, perform a training step."},
        {"load_merges", (PyCFunction) Trainer_load_merges, METH_VARARGS |
                                                           METH_KEYWORDS, "Load a \"merges\" list before training"},
        {NULL}  /* Sentinel */
};

static PyMethodDef Tokenizer_methods[] = {
        {"encode",       (PyCFunction) Tokenizer_encode,       METH_O,      "encode method (bytes -> ids)"},
        {"decode",       (PyCFunction) Tokenizer_decode,       METH_O,      "decode method (ids -> bytes)"},

        {"cache_decode", (PyCFunction) Tokenizer_cache_decode, METH_O,      "decode utf-8 bytes by one id"},
        {"cache_clean",  (PyCFunction) Tokenizer_cache_clean,  METH_NOARGS, "clean utf-8 bytes cache"},
        {NULL}  /* Sentinel */
};

static PyTypeObject TrainerType = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "bpe.Trainer",
        .tp_doc = PyDoc_STR("BPE Trainer"),
        .tp_basicsize = sizeof(TrainerObject),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) Trainer_init,
        .tp_dealloc = (destructor) Trainer_dealloc,
        .tp_getset = Trainer_getset,
        .tp_methods = Trainer_methods,
};

static PyTypeObject TokenizerType = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "bpe.Tokenizer",
        .tp_doc = PyDoc_STR("BPE Tokenizer"),
        .tp_basicsize = sizeof(TokenizerObject),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT,
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) Tokenizer_init,
        .tp_dealloc = (destructor) Tokenizer_dealloc,
        .tp_getset = Tokenizer_getset,
        .tp_methods = Tokenizer_methods,
};

static PyModuleDef bpe_module = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "bpe",
        .m_doc = "This is a Python-C-Extension module that implements the core algorithm of BPE (Byte-Pair-Encoding).",
        .m_size = -1,
};

PyMODINIT_FUNC PyInit_bpe(void) {
    if (PyType_Ready(&TrainerType) < 0 || PyType_Ready(&TokenizerType) < 0) {
        return NULL;
    }

    PyObject *m = PyModule_Create(&bpe_module);
    if (m == NULL) {
        return NULL;
    }

    if (PyModule_AddObjectRef(m, "Trainer", (PyObject *) &TrainerType) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddObjectRef(m, "Tokenizer", (PyObject *) &TokenizerType) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
