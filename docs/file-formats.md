# TinyBPE File Format Specification

## `.tinymodel` — BPE Model File

The `.tinymodel` format stores the merge pairs and optional byte remapping
table that define a BPE model. Files use the extension `.tinymodel`.

### Format v1 (current)

```
Line 1:   TinyBPE Model v<VERSION>
Line 2:   <remap_count>
Lines 3-N: [remap values, one per line]  (only if remap_count == 256)
Remaining: <left> <right>  (one merge pair per line)
```

**Header line**:
```
TinyBPE Model v1
```

**Remap count line**: Either `256` (byte remapping present) or `0` (no remapping).

**Remap values** (when count is `256`): 256 lines, each containing a single
integer in the range 0–255. Line `i` specifies the remapped value for input
byte `i`. When no remapping is needed, the identity mapping is implied
(byte `i` → byte `i`).

**Merge pairs**: Each subsequent line contains two space-separated integers
`<left> <right>`, representing a BPE merge pair. Pairs are listed in
**training order** (rank = 256 + line_index). The first pair has rank 256,
the second 257, etc.

### Example

```
TinyBPE Model v1
0
104 101
256 108
257 108
258 111
259 32
```

This represents a model with:
- No byte remapping (`0`)
- 5 merge pairs: `(104,101)` rank 256, `(256,108)` rank 257, etc.
- Vocabulary size: 256 + 5 = 261

### Example with byte remapping

```
TinyBPE Model v1
256
188
189
190
...
<th byte 255>
104 101
256 108
...
```

The 256 integers after the `256` header are the byte remapping table
(common in tiktoken-compatible models like cl100k_base).

### Legacy format (v0, deprecated)

```
Line 1:   TinyBPE Model
Line 2:   0
Remaining: <left> <right>
```

The legacy format differs only in the header: it lacks the `v<VERSION>`
suffix. The load function automatically detects and handles both formats.
New files are always written in v1 format.

### Validation Rules

When loading, the following checks are performed:

1. File extension must be `.tinymodel`.
2. Header must start with `TinyBPE Model`.
3. Version (if present) must not exceed the supported `MODEL_VERSION`.
4. Remap count must be either `0` or `256`.
5. If remap count is `256`, exactly 256 integers follow, each in 0–255.

---

## `.vocab` — Vocabulary File

The `.vocab` format stores the decoded vocabulary — the byte sequence
for each token ID. This format is **compatible with tiktoken**'s
`.tiktoken` file format. Files use the extension `.vocab`.

### Format v1 (current)

```
Line 1:       TinyBPE Vocabulary v<VERSION>
Remaining:    <base64_encoded_bytes> <rank>
```

**Header line**:
```
TinyBPE Vocabulary v1
```

**Data lines**: Each line contains a base64-encoded byte sequence followed
by a space and the integer token ID (rank). Lines are **sorted by rank**.

### Example

```
TinyBPE Vocabulary v1
AA== 0
AQ== 1
Ag== 2
Aw== 3
...
aGVsbG8= 259
d29ybGQ= 264
```

Decoded:
- Token 0: `b'\x00'` (base64 `AA==`)
- Token 1: `b'\x01'` (base64 `AQ==`)
- Token 259: `b'hello'` (base64 `aGVsbG8=`)
- Token 264: `b'world'` (base64 `d29ybGQ=`)

### Base64 Encoding

Token byte sequences are encoded using standard Base64 (RFC 4648).
Decoding is straightforward:

```python
import base64

# Read a .vocab file
vocab = {}
with open("model.vocab") as f:
    for line in f:
        if line.startswith("TinyBPE Vocabulary"):
            continue
        encoded, rank_str = line.strip().split()
        vocab[int(rank_str)] = base64.b64decode(encoded)
```

---

## File Relationships

```
Training
   │
   ▼
.tinymodel  ── merge pairs + optional byte remap
   │
   ├── Tokenizer(load_bpe_model("model.tinymodel"))
   │    │
   │    ├── .encode(text) → token IDs
   │    └── .decode(ids)  → text
   │
   └── .vocab  ── token ID → bytes mapping (derived from merges)
        │
        └── save_bpe_vocab("model", tokenizer.vocab)
```

The `.tinymodel` file is the **primary artifact** from training.
The `.vocab` file is a **derived artifact** for human inspection
and tiktoken compatibility. To reconstruct a tokenizer, only the
`.tinymodel` file is needed.

---

## Creating Model Files

### From training

```python
from tinybpe import SimpleTrainer

trainer = SimpleTrainer(text)
for _ in range(vocab_size - 256):
    trainer.step()
trainer.save("my-model")          # → my-model.tinymodel

# Optional: also save vocab
from tinybpe import Tokenizer, load_bpe_model
model = load_bpe_model("my-model.tinymodel")
tokenizer = Tokenizer(model)
tokenizer.save_vocab("my-model")  # → my-model.vocab
```

### From tiktoken

```python
from tinybpe import save_from_tiktoken
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
save_from_tiktoken("cl100k_base", enc._mergeable_ranks)
# → cl100k_base.tinymodel
```
