# TinyBPE File Formats

## `.tbm` — TinyBPE Model

The primary model file for a BPE tokenizer. Text format, human-readable.

### Format

```
TinyBPE Model v1
<remap_flag>
[256 lines of remap values if remap_flag = 256]
<left> <right>
<left> <right>
...
```

### Fields

| Line | Description |
|---|---|
| `TinyBPE Model v1` | Magic header with version |
| `remap_flag` | `0` (no byte remapping) or `256` (has remapping) |
| `remap_values` | 256 lines, each an integer 0-255. Present only if `remap_flag = 256` |
| `left right` | Merge pairs. `left` and `right` are unsigned integers. One pair per line |

### Example (no remapping)

```
TinyBPE Model v1
0
104 101
256 108
257 111
```

### Example (with remapping)

```
TinyBPE Model v1
256
42
0
1
...
104 101
256 108
```

---

## `.vocab` — Vocabulary File

Human-readable vocabulary file for inspection. Compatible with tiktoken's `.tiktoken` format.

### Format

```
TinyBPE Vocabulary v1
<base64_encoded_bytes> <rank>
<base64_encoded_bytes> <rank>
...
```

Each line is a base64-encoded byte sequence followed by its token ID (rank), separated by a space.

### Example

```
TinyBPE Vocabulary v1
aA== 65
aGVsbG8= 256
d29ybGQ= 257
```
