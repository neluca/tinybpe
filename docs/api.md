# TinyBPE API Reference

## `Tokenizer`

The main tokenizer class. Wraps the C-level BPE tokenizer with regex pre-tokenization, special token handling, byte remapping, and streaming decode.

```python
from tinybpe import Tokenizer
```

### Constructor

```python
Tokenizer(
    merges: list[tuple[int, int]],
    *,
    bytes_maps: list[int] | None = None,
    pat_str: str | None = None,
    special_tokens: dict[str, int] | None = None,
)
```

| Parameter | Description |
|---|---|
| `merges` | BPE merge pairs defining the vocabulary |
| `bytes_maps` | Optional byte remapping table (256 ints) for tiktoken compat |
| `pat_str` | Regex pattern for pre-tokenization. Default: `(?s)^.*$` (no split) |
| `special_tokens` | Dict mapping special token strings → their IDs |

### Methods

| Method | Description |
|---|---|
| `encode(text) → list[int]` | Encode text, respecting special tokens |
| `encode_ordinary(text) → list[int]` | Encode text, ignoring special tokens |
| `decode(ids) → str` | Decode token IDs back to text |
| `stream_decode(callback) → Callable[[int], None]` | Create streaming decoder |
| `stream_decode_reset()` | Clear streaming decode cache |
| `save(path)` | Save model to `.tbm` file |
| `save_vocab(path)` | Save vocabulary to `.vocab` file |

### Class Methods

| Method | Description |
|---|---|
| `from_file(path, *, pat_str=None, special_tokens=None) → Tokenizer` | Load from `.tbm` file |

### Properties

| Property | Type | Description |
|---|---|---|
| `merges` | `list[tuple[int, int]]` | BPE merge pairs |
| `vocab` | `dict[int, bytes]` | Token ID → byte sequence mapping |
| `n_vocab` | `int` | Total vocab size (256 + n_merges + n_special) |

---

## `Trainer`

BPE trainer extending the C-level `bpe.Trainer`.

```python
from tinybpe import Trainer
```

### Constructor

```python
Trainer(
    text: str,
    preprocess: Callable | None = None,
    *,
    callback: Callable | None = None,
)
```

| Parameter | Description |
|---|---|
| `text` | Training text |
| `preprocess` | Optional function `(str) → list[bytes\|bytearray]` for regex pre-tokenization |
| `callback` | Optional function `(step, total, pair, rank, freq)` called after each step |

### Methods

| Method | Description |
|---|---|
| `step() → tuple \| None` | Perform one training step. Returns `(pair, rank, frequency)` or `None` |
| `train(n) → int` | Train for `n` steps. Returns actual number performed |
| `load_merges(merges)` | Load existing merges for continue-training |
| `save(path)` | Save model to `.tbm` file |

### Properties

| Property | Type | Description |
|---|---|---|
| `merges` | `list[tuple[int, int]]` | Learned merge pairs |
| `n_merges` | `int` | Number of merges learned |

---

## File I/O

```python
from tinybpe import load_model, save_model, load_vocab, save_vocab
```

| Function | Description |
|---|---|
| `load_model(path) → tuple[list, list\|None]` | Load `.tbm` file, returns `(merges, bytes_maps)` |
| `save_model(path, merges, bytes_maps=None)` | Save `.tbm` file |
| `load_vocab(path) → dict[int, bytes]` | Load `.vocab` file |
| `save_vocab(path, vocab)` | Save `.vocab` file |
