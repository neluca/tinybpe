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

typedef struct {
    PyObject_HEAD

    unsigned char _map[256];
} BytesRemapObject;

static int trainer_init(TrainerObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"list_bytes", NULL};
    PyObject *list = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &list)) {
        return -1;
    }

    if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "The input argument must be a list containing bytes-like objects.");
        return -1;
    }

    Py_ssize_t list_len = PyList_Size(list);
    self->list_merges = NULL; // init

    if (list_len == 0) {
        PyErr_SetString(PyExc_Exception,
                        "The list must not be empty, and the objects in the list must be of bytes-like type.");
        return -1;
    }

    self->ctx.rank = BPE_TRAIN_ID_INIT;
    self->ctx.pieces_len = (size_t) list_len;
    self->ctx.pieces = bpe_malloc(list_len * sizeof(bpe_piece_t));

    for (Py_ssize_t i = 0; i < list_len; i++) {
        PyObject *item = PyList_GetItem(list, i);

        if (PyBytes_Check(item)) {
            Py_ssize_t size = PyBytes_Size(item);
            const char *bytes = PyBytes_AsString(item); // no free
            bpe_train_ctx_idx_init(&self->ctx, i, bytes, (size_t) size);
        }

        else if (PyByteArray_Check(item)) {
            Py_ssize_t size = PyByteArray_Size(item);
            const char *bytes = PyByteArray_AsString(item); // no free
            bpe_train_ctx_idx_init(&self->ctx, i, bytes, (size_t) size);
        }

        else {
            bpe_train_ctx_free(&self->ctx);
            bpe_free(self->ctx.pieces);
            self->ctx.pieces = NULL;
            PyErr_SetString(PyExc_TypeError, "The objects in the list must be of bytes-like type.");
            return -1;
        }
    }

    self->list_merges = PyList_New(0); // yes incref

    return 0;
}

static void trainer_dealloc(TrainerObject *self) {
    bpe_train_ctx_free(&self->ctx);
    bpe_free(self->ctx.pieces);

    Py_XDECREF(self->list_merges);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *trainer_get_merges(TrainerObject *self, void *Py_UNUSED(closure)) {
    return Py_NewRef(self->list_merges);  // yes incref
}

static PyObject *trainer_get_merges_size(TrainerObject *self, void *Py_UNUSED(closure)) {
    Py_ssize_t size = PyList_Size(self->list_merges);
    return PyLong_FromSsize_t(size); // yes incref
}

static PyObject *trainer_step(TrainerObject *self, PyObject *Py_UNUSED(args)) {
    bpe_pair_t pair;
    unsigned long count = bpe_get_max_count_pair(&pair, &self->ctx);

    if (count) {
        PyObject *pair_tuple = Py_BuildValue("(ii)", pair._1, pair._2); // yes incref
        PyList_Append(self->list_merges, pair_tuple); // yes incref

        return Py_BuildValue("(Oii)", pair_tuple, self->ctx.rank, count); // yes incref
    }

    return Py_None;
}

static PyObject *trainer_load_merges(TrainerObject *self, PyObject *args, PyObject *kwds) {
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
        PyErr_SetString(PyExc_TypeError, "The \"merges\" must be a list containing pairs.");
        return NULL;
    }

    Py_ssize_t merges_size = PyList_Size(list_merges);
    if (merges_size == 0) {
        PyErr_SetString(PyExc_ValueError, "The \"merges\" is a list with a non-zero length.");
        return NULL;
    }
    bpe_pair_t *pairs = bpe_malloc(merges_size * sizeof(bpe_pair_t));

    for (Py_ssize_t i = 0; i < merges_size; i++) {
        PyObject *item = PyList_GetItem(list_merges, i); // no incref
        PyObject *item_1 = PyTuple_GetItem(item, 0); // no incref
        PyObject *item_2 = PyTuple_GetItem(item, 1); // no incref

        pairs[i]._1 = PyLong_AsUnsignedLong(item_1);
        pairs[i]._2 = PyLong_AsUnsignedLong(item_2);

        if ((int) pairs[i]._1 < 0 || (int) pairs[i]._2 < 0) {
            bpe_free(pairs);
            PyErr_SetString(PyExc_ValueError, "The \"merges\" must be positive integer.");
            return NULL;
        }
    }

    if (!bpe_check(pairs, (size_t) merges_size)) {
        bpe_free(pairs);
        PyErr_SetString(PyExc_ValueError, "The provided \"merges\" is not valid.");
        return NULL;
    }

    Py_DECREF(self->list_merges);
    self->list_merges = list_merges;
    Py_INCREF(self->list_merges);

    bpe_apply_merges(&self->ctx, pairs, (size_t) merges_size);
    bpe_free(pairs);

    return Py_None;
}

static int tokenizer_init(TokenizerObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"merges", "special_tokens", NULL};
    PyObject *list_merges = NULL;
    PyObject *dict_special_tokens = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &list_merges, &dict_special_tokens)) {
        return -1;
    }

    if (!PyList_Check(list_merges)) {
        PyErr_SetString(PyExc_TypeError, "The \"merges\" must be a list containing pairs.");
        return -1;
    }

    Py_ssize_t merges_size = PyList_Size(list_merges);
    if (merges_size == 0) {
        PyErr_SetString(PyExc_Exception,
                        "The list must not be empty, and the objects in the list must be of tuple type.");
        return -1;
    }

    PyObject *tuple_item = PyList_GetItem(list_merges, 0); // no incref
    if (!PyTuple_Check(tuple_item) || PyTuple_Size(tuple_item) != 2) {
        PyErr_SetString(PyExc_TypeError,
                        "The objects in the list must be of tuple type, and the tuple must be pairs");
        return -1;
    }

    // init
    self->pairs = NULL;
    self->merges = NULL;
    self->vocab = NULL;
    self->list_merges = NULL;

    if (dict_special_tokens) {
        if (PyDict_Check(dict_special_tokens) && PyDict_Size(dict_special_tokens) != 0) {
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
            PyErr_WarnEx(PyExc_UserWarning, "special_tokens must be a dict with a non-zero length.", 1);
        }
    }
    else {
        self->dict_special_tokens = NULL;
        self->dict_inverse_special_tokens = NULL;
    }

    self->pairs_size = (size_t) merges_size;
    self->pairs = bpe_malloc(merges_size * sizeof(bpe_pair_t));

    for (Py_ssize_t i = 0; i < merges_size; i++) {
        PyObject *item = PyList_GetItem(list_merges, i); // no incref
        PyObject *item_1 = PyTuple_GetItem(item, 0); // no incref
        PyObject *item_2 = PyTuple_GetItem(item, 1); // no incref

        self->pairs[i]._1 = PyLong_AsUnsignedLong(item_1);
        self->pairs[i]._2 = PyLong_AsUnsignedLong(item_2);

        if ((int) self->pairs[i]._1 < 0 || (int) self->pairs[i]._2 < 0) {
            bpe_free(self->pairs); // Release in advance to avoid memory leaks.
            self->pairs = NULL;
            PyErr_SetString(PyExc_ValueError, "The pair of \"merges\" must be positive integer.");
            return -1;
        }
    }

    if (!bpe_check(self->pairs, self->pairs_size)) {
        bpe_free(self->pairs); // Release in advance to avoid memory leaks.
        self->pairs = NULL;
        PyErr_SetString(PyExc_ValueError, "The provided \"merges\" is not valid.");
        return -1;
    }

    self->list_merges = list_merges;
    Py_INCREF(self->list_merges);

    self->merges = bpe_merges_build(self->pairs, self->pairs_size);
    self->vocab = bpe_vocab_build(self->pairs, self->pairs_size);
    self->bytes_cache_size = 0;

    return 0;
}

static void tokenizer_dealloc(TokenizerObject *self) {
    bpe_free(self->pairs);
    self->pairs = NULL;
    bpe_merges_free(self->merges);
    self->merges = NULL;
    bpe_vocab_free(self->vocab);
    self->vocab = NULL;

    Py_XDECREF(self->list_merges);
    Py_XDECREF(self->dict_special_tokens);
    Py_XDECREF(self->dict_inverse_special_tokens);
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *tokenizer_get_merges(TokenizerObject *self, void *Py_UNUSED(closure)) {
    return Py_NewRef(self->list_merges); // yes incref
}

static PyObject *tokenizer_get_vocab(TokenizerObject *self, void *Py_UNUSED(closure)) {
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

static PyObject *tokenizer_get_size(TokenizerObject *self, void *Py_UNUSED(closure)) {
    Py_ssize_t dict_special_tokens_size = 0;
    if (self->dict_special_tokens) {
        dict_special_tokens_size = PyDict_Size(self->dict_special_tokens);
    }

    size_t size = self->vocab->vocab_size + (size_t) dict_special_tokens_size;
    return PyLong_FromSize_t(size);
}

static PyObject *tokenizer_encode(TokenizerObject *self, PyObject *bytes_o) {
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

    PyObject *ids_list = PyList_New((Py_ssize_t) ids_len); // yes incref
    for (size_t i = 0; i < ids_len; i++) {
        PyObject *id = PyLong_FromUnsignedLong(ids[i]); // yes incref
        PyList_SetItem(ids_list, (Py_ssize_t) i, id); // no incref
    }

    bpe_free(ids);

    return ids_list;
}

static PyObject *tokenizer_decode(TokenizerObject *self, PyObject *list_ids) {
    Py_ssize_t size = PyList_Size(list_ids);
    if (size == 0) {
        return PyBytes_FromString(""); // yes incref
    }

    unsigned long *ids = bpe_malloc(size * sizeof(unsigned long));
    size_t ids_size = 0;
    PyObject *bytes_obj = PyBytes_FromString(""); // yes incref

    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *item_id = PyList_GetItem(list_ids, i);
        unsigned long token_id = PyLong_AsLong(item_id);

        if (token_id >= self->vocab->vocab_size) {

            if (ids_size) {
                size_t bytes_size;
                char *c_bytes = bpe_decode(&bytes_size, self->vocab, ids, ids_size);

                PyBytes_Concat(&bytes_obj, PyBytes_FromStringAndSize(c_bytes, (Py_ssize_t) bytes_size)); // no incref

                bpe_free(c_bytes);
                ids_size = 0;
            }

            if (self->dict_inverse_special_tokens) {
                PyObject *special_bytes = PyDict_GetItem(self->dict_inverse_special_tokens, item_id); // no incref

                if (special_bytes) {
                    Py_INCREF(special_bytes);

                    PyBytes_Concat(&bytes_obj, special_bytes); // no incref
                }
                else {
                    PyErr_WarnFormat(PyExc_UserWarning, 1, "Unknown Token ID (%lu) \n", token_id);
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

        PyBytes_Concat(&bytes_obj, PyBytes_FromStringAndSize(c_bytes, (Py_ssize_t) bytes_size)); // no incref

        bpe_free(c_bytes);
    }

    bpe_free(ids);

    return bytes_obj;
}

static PyObject *tokenizer_cache_decode(TokenizerObject *self, PyObject *id_object) {
    if (self->bytes_cache_size && !bpe_utf8_head_check(self->bytes_cache[0])) {
        self->bytes_cache_size = 0;
    }
    unsigned long token_id = PyLong_AsLong(id_object);

    if (token_id < self->vocab->vocab_size) {
        size_t bytes_size;
        char *c_bytes = bpe_decode_one(&bytes_size, self->vocab, token_id, self->bytes_cache, &self->bytes_cache_size);
        PyObject *bytes_obj = Py_None;

        if (bytes_size) {
            bytes_obj = PyBytes_FromStringAndSize(c_bytes, (Py_ssize_t) bytes_size); // yes incref
        }

        bpe_free(c_bytes);

        return bytes_obj;
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
                PyErr_WarnFormat(PyExc_UserWarning, 1, "Unknown Token ID (%lu) \n", token_id);
            }
        }
        else {
            PyErr_WarnEx(PyExc_UserWarning, "No special_tokens.", 1);
        }
    }

    Py_RETURN_NONE;
}

static PyObject *tokenizer_cache_clean(TokenizerObject *self, PyObject *Py_UNUSED(args)) {
    self->bytes_cache_size = 0;
    Py_RETURN_NONE;
}

static int bytes_remap_init(BytesRemapObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"_remap", NULL};
    PyObject *list = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &list)) {
        return -1;
    }

    if (!PyList_Check(list) || PyList_Size(list) != 256) {
        return -1;
    }

    for (Py_ssize_t i = 0; i < 256; i++) {
        PyObject *item = PyList_GetItem(list, i);

        if (PyLong_Check(item)) {
            long item_long = PyLong_AsLong(item);
            if (item_long < 0 || item_long >= 256) {
                self->_map[i] = (unsigned char) item_long;
            }
            else {
                return -1;
            }
        }
        else {
            return -1;
        }
    }

    return 0;
}

static void bytes_remap_dealloc(BytesRemapObject *self) {
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *bytes_remap_call(BytesRemapObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"_bytes", NULL};
    const char *bytes = NULL;
    Py_ssize_t bytes_size;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "y#", kwlist, &bytes, &bytes_size)) {
        return NULL;
    }

    unsigned char *buf_bytes = bpe_malloc((size_t) bytes_size);
    for (Py_ssize_t i = 0; i < bytes_size; i++) {
        unsigned char map_i = (unsigned char) bytes[i];
        buf_bytes[i] = self->_map[map_i];
    }

    PyObject *bytes_obj = PyBytes_FromStringAndSize((const char *) buf_bytes, bytes_size);

    bpe_free(buf_bytes);

    return bytes_obj;
}

static PyGetSetDef trainer_getset[] = {
        {"merges",      (getter) trainer_get_merges,      NULL, "The \"merges\" list of the BPE algorithm", NULL},
        {"merges_size", (getter) trainer_get_merges_size, NULL, "The length of the \"merges\"",             NULL},
        {NULL}  /* Sentinel */
};

static PyGetSetDef tokenizer_getset[] = {
        {"merges", (getter) tokenizer_get_merges, NULL, "The \"merges\" list of the BPE algorithm", NULL},
        {"vocab",  (getter) tokenizer_get_vocab,  NULL, "Vocabulary dict of the BPE algorithm",     NULL},
        {"size",   (getter) tokenizer_get_size,   NULL, "The length of the vocabulary",             NULL},
        {NULL}  /* Sentinel */
};

static PyMethodDef trainer_methods[] = {
        {"step",        (PyCFunction) trainer_step,        METH_NOARGS, "During training, perform a training step."},
        {"load_merges", (PyCFunction) trainer_load_merges, METH_VARARGS |
                                                           METH_KEYWORDS, "Load a \"merges\" list before training"},
        {NULL}  /* Sentinel */
};

static PyMethodDef tokenizer_methods[] = {
        {"encode",       (PyCFunction) tokenizer_encode,       METH_O,      "encode method (bytes -> ids)"},
        {"decode",       (PyCFunction) tokenizer_decode,       METH_O,      "decode method (ids -> bytes)"},

        {"cache_decode", (PyCFunction) tokenizer_cache_decode, METH_O,      "decode utf-8 bytes by one id"},
        {"cache_clean",  (PyCFunction) tokenizer_cache_clean,  METH_NOARGS, "clean utf-8 bytes cache"},
        {NULL}  /* Sentinel */
};

// class Trainer
static PyTypeObject trainer_type = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "bpe.Trainer",
        .tp_doc = PyDoc_STR("BPE Trainer"),
        .tp_basicsize = sizeof(TrainerObject),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) trainer_init,
        .tp_dealloc = (destructor) trainer_dealloc,
        .tp_getset = trainer_getset,
        .tp_methods = trainer_methods,
};

// class Tokenizer
static PyTypeObject tokenizer_type = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "bpe.Tokenizer",
        .tp_doc = PyDoc_STR("BPE Tokenizer"),
        .tp_basicsize = sizeof(TokenizerObject),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT,
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) tokenizer_init,
        .tp_dealloc = (destructor) tokenizer_dealloc,
        .tp_getset = tokenizer_getset,
        .tp_methods = tokenizer_methods,
};

// class BytesRemap
static PyTypeObject bytes_remap_type = {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "bpe.BytesRemap",
        .tp_doc = PyDoc_STR("BytesRemap"),
        .tp_basicsize = sizeof(BytesRemapObject),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT,
        .tp_new = PyType_GenericNew,
        .tp_init = (initproc) bytes_remap_init,
        .tp_dealloc = (destructor) bytes_remap_dealloc,
        .tp_call = (ternaryfunc) bytes_remap_call,
};

static PyModuleDef bpe_module = {
        .m_base = PyModuleDef_HEAD_INIT,
        .m_name = "bpe",
        .m_doc = "This is a Python-C-Extension module that implements the core algorithm of BPE (Byte-Pair-Encoding).",
        .m_size = -1,
};

PyMODINIT_FUNC PyInit_bpe(void) {
    // check
    if (PyType_Ready(&trainer_type) < 0
        || PyType_Ready(&tokenizer_type) < 0
        || PyType_Ready(&bytes_remap_type) < 0) {
        return NULL;
    }

    // create module
    PyObject *m = PyModule_Create(&bpe_module);
    if (m == NULL) {
        return NULL;
    }

    if (PyModule_AddObjectRef(m, "Trainer", (PyObject *) &trainer_type) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddObjectRef(m, "Tokenizer", (PyObject *) &tokenizer_type) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddObjectRef(m, "BytesRemap", (PyObject *) &bytes_remap_type) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
