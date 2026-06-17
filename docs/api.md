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
| `encode_ordinary(text) → list[int]` | Encode text, ignoring special token pattern matching |
| `count_tokens(text) → int` | Return the number of tokens `text` would produce (convenience, same as `len(encode(text))`) |
| `decode(ids) → str` | Decode token IDs back to text |
| `stream_decode(callback) → Callable[[int], None]` | Create a streaming decoder. The returned callable accepts one token ID at a time; each complete text fragment is passed to `callback` |
| `stream_decode_reset()` | Clear streaming decode cache (for reuse) |
| `save(path)` | Save model to `.tbm` file |
| `save_vocab(path)` | Save vocabulary to `.vocab` file |

### Class Methods

| Method | Description |
|---|---|
| `from_file(path, *, pat_str=None, special_tokens=None) → Tokenizer` | Load from `.tbm` file |
| `from_pretrained(name) → Tokenizer` | Load a built-in model by name (e.g. `"cl100k_base"`). No network required — models ship with the package |

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

## Model Discovery

```python
from tinybpe import list_models, get_model_info
```

| Function | Description |
|---|---|
| `list_models() → list[str]` | Return sorted list of all built-in model names |
| `get_model_info(name) → dict` | Return metadata dict for a built-in model with keys: `name`, `path`, `vocab_size`, `description`, `family`, `pat_str`, `special_tokens`, `has_byte_remap` |

### Example

```python
>>> import tinybpe
>>> tinybpe.list_models()
['cl100k_base', 'deepseek-v4', 'llama4', 'minicpm5', 'o200k_base', 'p50k_base', 'qwen35', 'r50k_base']

>>> info = tinybpe.get_model_info("cl100k_base")
>>> info["vocab_size"]
100277
>>> info["family"]
'GPT-4'
```

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
