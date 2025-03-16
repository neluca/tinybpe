/*
 * Copyright Yinan Liao. and other contributors. All rights reserved.
 *
 * This code is adapted from "tree.h" of libuv, which is an elegant
 * implementation of a red-black tree, and its LICENSE is at the end
 * of the file.
 *
 * Reference: https://github.com/libuv/libuv/blob/v1.x/include/uv/tree.h
 */

#ifndef MICRO_RBTREE_H
#define MICRO_RBTREE_H

/*
 * This file defines data structures for red-black trees.
 * A red-black tree is a binary search tree with the node color as an
 * extra attribute.  It fulfills a set of conditions:
 *  - every search path from the root to a leaf consists of the
 *    same number of black nodes,
 *  - each red node (except for the root) has a black parent,
 *  - each leaf node is black.
 *
 * Every operation on a red-black tree is bounded as O(lg n).
 * The maximum height of a red-black tree is 2lg (n+1).
 */

/* Macros that define a red-black tree */
#define RB_HEAD(name, type)                                                   \
struct name {                                                                 \
  struct type *rbh_root; /* root of the tree */                               \
}

#define RB_INITIALIZER { NULL }
#define RB_BLACK  0
#define RB_RED    1

#define RB_ENTRY(type)                                                        \
struct {                                                                      \
  struct type *rbe_left;        /* left element */                            \
  struct type *rbe_right;       /* right element */                           \
  struct type *rbe_parent;      /* parent element */                          \
  int rbe_color;                /* node color */                              \
}

#define RB_LEFT(elm, field)     (elm)->field.rbe_left
#define RB_RIGHT(elm, field)    (elm)->field.rbe_right
#define RB_PARENT(elm, field)   (elm)->field.rbe_parent
#define RB_COLOR(elm, field)    (elm)->field.rbe_color
#define RB_ROOT(head)           (head)->rbh_root

#define RB_SET(elm, parent, field) do {                                       \
  RB_PARENT(elm, field) = parent;                                             \
  RB_LEFT(elm, field) = RB_RIGHT(elm, field) = NULL;                          \
  RB_COLOR(elm, field) = RB_RED;                                              \
} while (/*CONSTCOND*/ 0)

#define RB_SET_BLACKRED(black, red, field) do {                               \
  RB_COLOR(black, field) = RB_BLACK;                                          \
  RB_COLOR(red, field) = RB_RED;                                              \
} while (/*CONSTCOND*/ 0)

#define RB_ROTATE_LEFT(head, elm, tmp, field) do {                            \
  (tmp) = RB_RIGHT(elm, field);                                               \
  if ((RB_RIGHT(elm, field) = RB_LEFT(tmp, field)) != NULL) {                 \
    RB_PARENT(RB_LEFT(tmp, field), field) = (elm);                            \
  }                                                                           \
  if ((RB_PARENT(tmp, field) = RB_PARENT(elm, field)) != NULL) {              \
    if ((elm) == RB_LEFT(RB_PARENT(elm, field), field))                       \
      RB_LEFT(RB_PARENT(elm, field), field) = (tmp);                          \
    else                                                                      \
      RB_RIGHT(RB_PARENT(elm, field), field) = (tmp);                         \
  } else                                                                      \
    (head)->rbh_root = (tmp);                                                 \
  RB_LEFT(tmp, field) = (elm);                                                \
  RB_PARENT(elm, field) = (tmp);                                              \
} while (/*CONSTCOND*/ 0)

#define RB_ROTATE_RIGHT(head, elm, tmp, field) do {                           \
  (tmp) = RB_LEFT(elm, field);                                                \
  if ((RB_LEFT(elm, field) = RB_RIGHT(tmp, field)) != NULL) {                 \
    RB_PARENT(RB_RIGHT(tmp, field), field) = (elm);                           \
  }                                                                           \
  if ((RB_PARENT(tmp, field) = RB_PARENT(elm, field)) != NULL) {              \
    if ((elm) == RB_LEFT(RB_PARENT(elm, field), field))                       \
      RB_LEFT(RB_PARENT(elm, field), field) = (tmp);                          \
    else                                                                      \
      RB_RIGHT(RB_PARENT(elm, field), field) = (tmp);                         \
  } else                                                                      \
    (head)->rbh_root = (tmp);                                                 \
  RB_RIGHT(tmp, field) = (elm);                                               \
  RB_PARENT(elm, field) = (tmp);                                              \
} while (/*CONSTCOND*/ 0)

/* Generates functions for red-black tree operations. */
#if __GNUC__
# define  RB_GENERATE(name, type, field, cmp)                                 \
  RB_GENERATE_INTERNAL(name, type, field, cmp, __attribute__((unused)) static)
#else
# define  RB_GENERATE(name, type, field, cmp)                                 \
  RB_GENERATE_INTERNAL(name, type, field, cmp, static)
#endif

#define RB_GENERATE_INTERNAL(name, type, field, cmp, attr)                    \
attr void                                                                     \
name##_RB_INSERT_COLOR(struct name *head, struct type *elm)                   \
{                                                                             \
  struct type *parent, *gparent, *tmp;                                        \
  while ((parent = RB_PARENT(elm, field)) != NULL &&                          \
      RB_COLOR(parent, field) == RB_RED) {                                    \
    gparent = RB_PARENT(parent, field);                                       \
    if (parent == RB_LEFT(gparent, field)) {                                  \
      tmp = RB_RIGHT(gparent, field);                                         \
      if (tmp && RB_COLOR(tmp, field) == RB_RED) {                            \
        RB_COLOR(tmp, field) = RB_BLACK;                                      \
        RB_SET_BLACKRED(parent, gparent, field);                              \
        elm = gparent;                                                        \
        continue;                                                             \
      }                                                                       \
      if (RB_RIGHT(parent, field) == elm) {                                   \
        RB_ROTATE_LEFT(head, parent, tmp, field);                             \
        tmp = parent;                                                         \
        parent = elm;                                                         \
        elm = tmp;                                                            \
      }                                                                       \
      RB_SET_BLACKRED(parent, gparent, field);                                \
      RB_ROTATE_RIGHT(head, gparent, tmp, field);                             \
    } else {                                                                  \
      tmp = RB_LEFT(gparent, field);                                          \
      if (tmp && RB_COLOR(tmp, field) == RB_RED) {                            \
        RB_COLOR(tmp, field) = RB_BLACK;                                      \
        RB_SET_BLACKRED(parent, gparent, field);                              \
        elm = gparent;                                                        \
        continue;                                                             \
      }                                                                       \
      if (RB_LEFT(parent, field) == elm) {                                    \
        RB_ROTATE_RIGHT(head, parent, tmp, field);                            \
        tmp = parent;                                                         \
        parent = elm;                                                         \
        elm = tmp;                                                            \
      }                                                                       \
      RB_SET_BLACKRED(parent, gparent, field);                                \
      RB_ROTATE_LEFT(head, gparent, tmp, field);                              \
    }                                                                         \
  }                                                                           \
  RB_COLOR(head->rbh_root, field) = RB_BLACK;                                 \
}                                                                             \
                                                                              \
attr void                                                                     \
name##_RB_REMOVE_COLOR(struct name *head, struct type *parent,                \
    struct type *elm)                                                         \
{                                                                             \
  struct type *tmp;                                                           \
  while ((elm == NULL || RB_COLOR(elm, field) == RB_BLACK) &&                 \
      elm != RB_ROOT(head)) {                                                 \
    if (RB_LEFT(parent, field) == elm) {                                      \
      tmp = RB_RIGHT(parent, field);                                          \
      if (RB_COLOR(tmp, field) == RB_RED) {                                   \
        RB_SET_BLACKRED(tmp, parent, field);                                  \
        RB_ROTATE_LEFT(head, parent, tmp, field);                             \
        tmp = RB_RIGHT(parent, field);                                        \
      }                                                                       \
      if ((RB_LEFT(tmp, field) == NULL ||                                     \
          RB_COLOR(RB_LEFT(tmp, field), field) == RB_BLACK) &&                \
          (RB_RIGHT(tmp, field) == NULL ||                                    \
          RB_COLOR(RB_RIGHT(tmp, field), field) == RB_BLACK)) {               \
        RB_COLOR(tmp, field) = RB_RED;                                        \
        elm = parent;                                                         \
        parent = RB_PARENT(elm, field);                                       \
      } else {                                                                \
        if (RB_RIGHT(tmp, field) == NULL ||                                   \
            RB_COLOR(RB_RIGHT(tmp, field), field) == RB_BLACK) {              \
          struct type *oleft;                                                 \
          if ((oleft = RB_LEFT(tmp, field))                                   \
              != NULL)                                                        \
            RB_COLOR(oleft, field) = RB_BLACK;                                \
          RB_COLOR(tmp, field) = RB_RED;                                      \
          RB_ROTATE_RIGHT(head, tmp, oleft, field);                           \
          tmp = RB_RIGHT(parent, field);                                      \
        }                                                                     \
        RB_COLOR(tmp, field) = RB_COLOR(parent, field);                       \
        RB_COLOR(parent, field) = RB_BLACK;                                   \
        if (RB_RIGHT(tmp, field))                                             \
          RB_COLOR(RB_RIGHT(tmp, field), field) = RB_BLACK;                   \
        RB_ROTATE_LEFT(head, parent, tmp, field);                             \
        elm = RB_ROOT(head);                                                  \
        break;                                                                \
      }                                                                       \
    } else {                                                                  \
      tmp = RB_LEFT(parent, field);                                           \
      if (RB_COLOR(tmp, field) == RB_RED) {                                   \
        RB_SET_BLACKRED(tmp, parent, field);                                  \
        RB_ROTATE_RIGHT(head, parent, tmp, field);                            \
        tmp = RB_LEFT(parent, field);                                         \
      }                                                                       \
      if ((RB_LEFT(tmp, field) == NULL ||                                     \
          RB_COLOR(RB_LEFT(tmp, field), field) == RB_BLACK) &&                \
          (RB_RIGHT(tmp, field) == NULL ||                                    \
          RB_COLOR(RB_RIGHT(tmp, field), field) == RB_BLACK)) {               \
        RB_COLOR(tmp, field) = RB_RED;                                        \
        elm = parent;                                                         \
        parent = RB_PARENT(elm, field);                                       \
      } else {                                                                \
        if (RB_LEFT(tmp, field) == NULL ||                                    \
            RB_COLOR(RB_LEFT(tmp, field), field) == RB_BLACK) {               \
          struct type *oright;                                                \
          if ((oright = RB_RIGHT(tmp, field))                                 \
              != NULL)                                                        \
            RB_COLOR(oright, field) = RB_BLACK;                               \
          RB_COLOR(tmp, field) = RB_RED;                                      \
          RB_ROTATE_LEFT(head, tmp, oright, field);                           \
          tmp = RB_LEFT(parent, field);                                       \
        }                                                                     \
        RB_COLOR(tmp, field) = RB_COLOR(parent, field);                       \
        RB_COLOR(parent, field) = RB_BLACK;                                   \
        if (RB_LEFT(tmp, field))                                              \
          RB_COLOR(RB_LEFT(tmp, field), field) = RB_BLACK;                    \
        RB_ROTATE_RIGHT(head, parent, tmp, field);                            \
        elm = RB_ROOT(head);                                                  \
        break;                                                                \
      }                                                                       \
    }                                                                         \
  }                                                                           \
  if (elm)                                                                    \
    RB_COLOR(elm, field) = RB_BLACK;                                          \
}                                                                             \
                                                                              \
attr struct type *                                                            \
name##_RB_REMOVE(struct name *head, struct type *elm)                         \
{                                                                             \
  struct type *child, *parent, *old = elm;                                    \
  int color;                                                                  \
  if (RB_LEFT(elm, field) == NULL)                                            \
    child = RB_RIGHT(elm, field);                                             \
  else if (RB_RIGHT(elm, field) == NULL)                                      \
    child = RB_LEFT(elm, field);                                              \
  else {                                                                      \
    struct type *left;                                                        \
    elm = RB_RIGHT(elm, field);                                               \
    while ((left = RB_LEFT(elm, field)) != NULL)                              \
      elm = left;                                                             \
    child = RB_RIGHT(elm, field);                                             \
    parent = RB_PARENT(elm, field);                                           \
    color = RB_COLOR(elm, field);                                             \
    if (child)                                                                \
      RB_PARENT(child, field) = parent;                                       \
    if (parent) {                                                             \
      if (RB_LEFT(parent, field) == elm)                                      \
        RB_LEFT(parent, field) = child;                                       \
      else                                                                    \
        RB_RIGHT(parent, field) = child;                                      \
    } else                                                                    \
      RB_ROOT(head) = child;                                                  \
    if (RB_PARENT(elm, field) == old)                                         \
      parent = elm;                                                           \
    (elm)->field = (old)->field;                                              \
    if (RB_PARENT(old, field)) {                                              \
      if (RB_LEFT(RB_PARENT(old, field), field) == old)                       \
        RB_LEFT(RB_PARENT(old, field), field) = elm;                          \
      else                                                                    \
        RB_RIGHT(RB_PARENT(old, field), field) = elm;                         \
    } else                                                                    \
      RB_ROOT(head) = elm;                                                    \
    RB_PARENT(RB_LEFT(old, field), field) = elm;                              \
    if (RB_RIGHT(old, field))                                                 \
      RB_PARENT(RB_RIGHT(old, field), field) = elm;                           \
    if (parent) {                                                             \
      left = parent;                                                          \
      do {                                                                    \
      } while ((left = RB_PARENT(left, field)) != NULL);                      \
    }                                                                         \
    goto color;                                                               \
  }                                                                           \
  parent = RB_PARENT(elm, field);                                             \
  color = RB_COLOR(elm, field);                                               \
  if (child)                                                                  \
    RB_PARENT(child, field) = parent;                                         \
  if (parent) {                                                               \
    if (RB_LEFT(parent, field) == elm)                                        \
      RB_LEFT(parent, field) = child;                                         \
    else                                                                      \
      RB_RIGHT(parent, field) = child;                                        \
  } else                                                                      \
    RB_ROOT(head) = child;                                                    \
color:                                                                        \
  if (color == RB_BLACK)                                                      \
    name##_RB_REMOVE_COLOR(head, parent, child);                              \
  return (old);                                                               \
}                                                                             \
                                                                              \
/* Inserts a node into the RB tree */                                         \
attr struct type *                                                            \
name##_RB_INSERT(struct name *head, struct type *elm)                         \
{                                                                             \
  struct type *tmp;                                                           \
  struct type *parent = NULL;                                                 \
  int comp = 0;                                                               \
  tmp = RB_ROOT(head);                                                        \
  while (tmp) {                                                               \
    parent = tmp;                                                             \
    comp = (cmp)(elm, parent);                                                \
    if (comp < 0)                                                             \
      tmp = RB_LEFT(tmp, field);                                              \
    else if (comp > 0)                                                        \
      tmp = RB_RIGHT(tmp, field);                                             \
    else                                                                      \
      return (tmp);                                                           \
  }                                                                           \
  RB_SET(elm, parent, field);                                                 \
  if (parent != NULL) {                                                       \
    if (comp < 0)                                                             \
      RB_LEFT(parent, field) = elm;                                           \
    else                                                                      \
      RB_RIGHT(parent, field) = elm;                                          \
  } else                                                                      \
    RB_ROOT(head) = elm;                                                      \
  name##_RB_INSERT_COLOR(head, elm);                                          \
  return (NULL);                                                              \
}                                                                             \
                                                                              \
/* Finds the node with the same key as elm */                                 \
attr struct type *                                                            \
name##_RB_FIND(struct name *head, struct type *elm)                           \
{                                                                             \
  struct type *tmp = RB_ROOT(head);                                           \
  int comp;                                                                   \
  while (tmp) {                                                               \
    comp = cmp(elm, tmp);                                                     \
    if (comp < 0)                                                             \
      tmp = RB_LEFT(tmp, field);                                              \
    else if (comp > 0)                                                        \
      tmp = RB_RIGHT(tmp, field);                                             \
    else                                                                      \
      return (tmp);                                                           \
  }                                                                           \
  return (NULL);                                                              \
}                                                                             \
                                                                              \
/* ARGSUSED */                                                                \
attr struct type *                                                            \
name##_RB_NEXT(struct type *elm)                                              \
{                                                                             \
  if (RB_RIGHT(elm, field)) {                                                 \
    elm = RB_RIGHT(elm, field);                                               \
    while (RB_LEFT(elm, field))                                               \
      elm = RB_LEFT(elm, field);                                              \
  } else {                                                                    \
    if (RB_PARENT(elm, field) &&                                              \
        (elm == RB_LEFT(RB_PARENT(elm, field), field)))                       \
      elm = RB_PARENT(elm, field);                                            \
    else {                                                                    \
      while (RB_PARENT(elm, field) &&                                         \
          (elm == RB_RIGHT(RB_PARENT(elm, field), field)))                    \
        elm = RB_PARENT(elm, field);                                          \
      elm = RB_PARENT(elm, field);                                            \
    }                                                                         \
  }                                                                           \
  return (elm);                                                               \
}                                                                             \
                                                                              \
/* ARGSUSED */                                                                \
attr struct type *                                                            \
name##_RB_PREV(struct type *elm)                                              \
{                                                                             \
  if (RB_LEFT(elm, field)) {                                                  \
    elm = RB_LEFT(elm, field);                                                \
    while (RB_RIGHT(elm, field))                                              \
      elm = RB_RIGHT(elm, field);                                             \
  } else {                                                                    \
    if (RB_PARENT(elm, field) &&                                              \
        (elm == RB_RIGHT(RB_PARENT(elm, field), field)))                      \
      elm = RB_PARENT(elm, field);                                            \
    else {                                                                    \
      while (RB_PARENT(elm, field) &&                                         \
          (elm == RB_LEFT(RB_PARENT(elm, field), field)))                     \
        elm = RB_PARENT(elm, field);                                          \
      elm = RB_PARENT(elm, field);                                            \
    }                                                                         \
  }                                                                           \
  return (elm);                                                               \
}                                                                             \
                                                                              \
attr struct type *                                                            \
name##_RB_MINMAX(struct name *head, int val)                                  \
{                                                                             \
  struct type *tmp = RB_ROOT(head);                                           \
  struct type *parent = NULL;                                                 \
  while (tmp) {                                                               \
    parent = tmp;                                                             \
    if (val < 0)                                                              \
      tmp = RB_LEFT(tmp, field);                                              \
    else                                                                      \
      tmp = RB_RIGHT(tmp, field);                                             \
  }                                                                           \
  return (parent);                                                            \
}

#define RB_LT   (-1)
#define RB_GT     1

#define RB_INSERT(name, head, elm)   name##_RB_INSERT(head, elm)
#define RB_REMOVE(name, head, elm)   name##_RB_REMOVE(head, elm)
#define RB_FIND(name, head, elm)     name##_RB_FIND(head, elm)
#define RB_NEXT(name, elm)           name##_RB_NEXT(elm)
#define RB_PREV(name, elm)           name##_RB_PREV(elm)
#define RB_MIN(name, head)           name##_RB_MINMAX(head, RB_LT)
#define RB_MAX(name, head)           name##_RB_MINMAX(head, RB_GT)

#define RB_FOREACH(x, name, head)                                             \
  for ((x) = RB_MIN(name, head);                                              \
       (x) != NULL;                                                           \
       (x) = RB_NEXT(name, x))

#define RB_FOREACH_REVERSE(x, name, head)                                     \
  for ((x) = RB_MAX(name, head);                                              \
       (x) != NULL;                                                           \
       (x) = RB_PREV(name, x))

#endif  /* MICRO_RBTREE_H */

/*-
 * Copyright 2002 Niels Provos <provos@citi.umich.edu>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */