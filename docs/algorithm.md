# BPE Algorithm &amp; Implementation

## Byte Pair Encoding (BPE) Overview

BPE is a data compression algorithm adapted for NLP tokenization. Starting
from the 256 base byte tokens (0–255), it iteratively finds the **most
frequent adjacent pair** of tokens in the training corpus and creates a
new token for that pair. After `N` merges, the vocabulary size is `256 + N`.

The core insight: frequently co-occurring byte sequences (like `"th"`, `"he"`,
`"ing"`) get merged into single tokens, building a vocabulary that captures
common subword patterns.

### Training Algorithm

```
Input:  training text (as bytes)
Output: ordered list of merge pairs [(left, right), ...]

1. Split training text into pieces (e.g., by regex pattern or newlines).
2. Convert each byte to a token ID (0–255). Each piece is now a sequence of token IDs.
3. Repeat until desired vocab size or no more pairs:
   a. Count all adjacent pairs across all pieces.
   b. Find the pair with the highest frequency.
   c. If max frequency > 0:
        Assign the next available token ID (starting at 256).
        Replace all occurrences of the pair with the new ID in-place.
        Record the merge pair.
   d. Else: stop (no more pairs available).
```

### Encoding Algorithm (Greedy Lowest-Rank-First)

Given a trained merge list, encoding converts input bytes to token IDs:

```
Input:  bytes, merges (ordered list of pairs)
Output: list of token IDs

1. Convert each byte to a base token ID (0–255).
2. While len(ids) > 1:
   a. For each adjacent pair, look up its merge rank in the merges tree.
      (Rank = index in merge list + 256. Lower rank = merged earlier.)
   b. Find the pair with the lowest rank.
   c. If no pair has a valid rank: stop.
   d. Replace all occurrences of the lowest-rank pair with its merged token ID.
```

This greedy approach ensures **deterministic, reproducible** encoding: the
same merges always produce the same token IDs.

### Decoding

Decoding is trivial O(n): each token ID maps directly to its byte sequence
via the vocabulary array. Concatenation of all token byte sequences
reconstructs the original bytes.

---

## Data Structures

### AVL Tree (for pair lookups)

Both training and encoding use AVL trees for O(log n) pair lookup:

- **Training**: An AVL tree tracks pair frequencies. Each node stores
  a `(left, right)` pair and its occurrence count. New pairs are inserted;
  existing pairs have their count incremented. After counting, a linear
  scan over the inserted nodes finds the maximum.

- **Encoding**: An AVL tree maps `(left, right)` pair → merge rank.
  During each encoding iteration, every adjacent pair in the current
  token sequence is looked up in the tree. The pair with the lowest
  rank is selected for merging.

**Memory optimization**: Each node's 2-bit balance factor (-1, 0, +1)
is packed into the low 2 bits of the parent pointer. This avoids an
extra `int` field per node. Extraction macros:

```c
#define avl_parent(node) ((struct avl_node *)((uintptr_t)(node)->parent & ~0x3))
#define avl_bf(node)     (((int)((uintptr_t)(node)->parent & 0x3)) - 1)
```

### Vocabulary (flat array)

The vocabulary is a flat array indexed by token ID, providing O(1) decode:

```
tokens[0]:   b'\x00'   (1 byte)
tokens[1]:   b'\x01'   (1 byte)
...
tokens[255]: b'\xff'   (1 byte)
tokens[256]: b'he'     (2 bytes — first merge)
tokens[257]: b'hel'    (3 bytes)
...
```

All token byte sequences are stored in a **single contiguous allocation**
(`bytes_mem`) for cache efficiency. Each `tokens[i].bytes` pointer points
into this allocation, and `tokens[i].size` stores the byte length.

**Construction**: Starting from the 256 single-byte tokens, each merge
pair `(left, right)` creates a new token whose bytes are the concatenation
of `vocab[left]` + `vocab[right]`. A two-pass approach is used:

1. **Pass 1**: Calculate sizes of all merged tokens (needed to allocate
   the contiguous memory block).
2. **Pass 2**: Populate the byte sequences and set pointers.

---

## Streaming Decode

`bpe_decode_one()` decodes one token ID at a time, enabling real-time
streaming applications (e.g., displaying LLM output token-by-token).

**Problem**: Multi-byte UTF-8 characters (2–4 bytes) may span token
boundaries. If a token ends with incomplete UTF-8 bytes, they must be
held until the next token provides the remaining bytes.

**Solution**: A 4-byte internal cache stores partial UTF-8 sequences.
Each call:

1. Appends the new token's bytes to the cache.
2. Walks the buffer using `bpe_utf8_length_from_head()` to detect
   complete UTF-8 character boundaries.
3. Returns completed characters and retains incomplete bytes.
4. Treats continuation bytes (0x80–0xBF) and invalid bytes as
   1-byte fragments to ensure forward progress.

---

## UTF-8 Byte Length Detection

```c
static inline int bpe_utf8_length_from_head(unsigned char head_byte) {
    if ((head_byte & 0x80) == 0)      return 1;  // 0xxxxxxx  (ASCII)
    else if ((head_byte & 0xE0) == 0xC0) return 2; // 110xxxxx  (2-byte)
    else if ((head_byte & 0xF0) == 0xE0) return 3; // 1110xxxx  (3-byte)
    else if ((head_byte & 0xF8) == 0xF0) return 4; // 11110xxx  (4-byte)
    return 0;  // continuation byte (10xxxxxx) or invalid (11111xxx)
}
```

Valid UTF-8 per RFC 3629 uses at most 4 bytes (U+0000 to U+10FFFF).
Bytes 0xF5–0xFF are invalid start bytes and return 0.

---

## Time &amp; Space Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Training step | O(P log U) | P = total adjacent pairs, U = unique pairs |
| Encode | O(L·W log V) | L = input length, W = merge passes, V = vocab size |
| Decode (batch) | O(N) | N = number of token IDs |
| Decode (stream) | O(K) per token | K = token byte length |
| Vocab lookup | O(1) | Flat array index |

In practice, `W` (number of merge passes) is small because each pass
reduces the sequence length proportionally to the pair frequency.
