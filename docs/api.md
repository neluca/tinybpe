# TinyBPE API Reference

TinyBPE provides a clean Python API for BPE tokenization and training.
The core algorithm runs in a CPython extension (`tinybpe.bpe`); the Python
layer adds text handling, regex pre-tokenization, file I/O, and tiktoken
compatibility.

## Tokenizer Classes

### `CommonTokenizer`

A byte-level BPE tokenizer with regex pre-tokenization support. This is the
simpler variant **without** byte remapping. For tiktoken compatibility, use
`Tokenizer` instead.

```python
from tinybpe import CommonTokenizer

tokenizer = CommonTokenizer(merges, pat_str=None, special_tokens=None)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `merges` | `list[tuple[int, int]]` | BPE merge pairs defining the vocabulary. |
| `pat_str` | `str \| None` | Regex pattern for pre-tokenization. Defaults to `r"^.*$"` (no splitting). |
| `special_tokens` | `dict[str, int] \| None` | Mapping from special token strings to their IDs. |

**Methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `encode` | `(text: str) -> list[int]` | Encode text to token IDs, including special tokens. |
| `encode_ordinary` | `(text: str) -> list[int]` | Encode text to token IDs, ignoring special tokens. |
| `decode` | `(ids: list[int]) -> str` | Decode token IDs back to a string. |
| `stream_decode` | `(callback: Callable[[str], None]) -> Callable[[int], None]` | Create a streaming decoder. Returns a function that accepts one token ID at a time and calls `callback` with each decoded text fragment. |
| `stream_decode_cache_clean` | `() -> None` | Clear the internal cache used by streaming decode. |
| `save` | `(file_prefix: str) -> None` | Save model to `<file_prefix>.tinymodel`. Only merge pairs are saved; regex pattern and special tokens are NOT preserved. |
| `save_vocab` | `(file_prefix: str) -> None` | Save vocabulary to `<file_prefix>.vocab`. |

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `merges` | `list[tuple[int, int]]` | The BPE merge pairs. |
| `vocab` | `dict[int, bytes]` | Vocabulary mapping token IDs to byte sequences. |
| `n_vocab` | `int` | Total vocabulary size (`256 + n_merges + n_special_tokens`). |

---

### `Tokenizer(CommonTokenizer)`

A BPE tokenizer **with byte remapping support**, necessary for tiktoken
compatibility. Byte remapping permutes byte values before encoding and
after decoding — some tiktoken models use non-identity byte-to-token
mappings.

```python
from tinybpe import Tokenizer, load_bpe_model

model = load_bpe_model("path/to/model.tinymodel")
tokenizer = Tokenizer(model, pat_str=None, special_tokens=None)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `bpe_param` | `BPEParam` | Model parameters including merges and optional byte remapping. |
| `pat_str` | `str \| None` | Regex pattern for pre-tokenization. |
| `special_tokens` | `dict[str, int] \| None` | Mapping from special token strings to their IDs. |

Inherits all methods and properties from `CommonTokenizer`. Additionally:

- **`encode_ordinary`** applies forward byte remapping before encoding.
- **`decode`** applies inverse byte remapping after decoding.
- **`stream_decode`** uses Python-level UTF-8 boundary caching with remap.
- **`save`** preserves byte remapping in the `.tinymodel` file.

---

## Trainer

### `SimpleTrainer`

A BPE trainer wrapping the C-level `bpe.Trainer` with automatic UTF-8
text encoding and optional preprocessing.

```python
from tinybpe import SimpleTrainer

trainer = SimpleTrainer(text, preprocess=None, callback=None)
```

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `text` | `str` | Training text (UTF-8 encoded if no preprocess is given). |
| `preprocess` | `Callable[[str], list[bytes \| bytearray]] \| None` | Optional preprocessing function. Use for regex pre-tokenization of training data. |
| `callback` | `Callable[[int, int, tuple[int, int], int, int], None] \| None` | Optional progress callback receiving `(step, total, pair, rank, frequency)`. |

**Methods**

| Method | Signature | Description |
|--------|-----------|-------------|
| `step` | `() -> tuple[tuple[int,int], int, int] \| None` | Perform one BPE training step. Returns `(pair, rank, frequency)` or `None` if no more merges possible. |
| `train` | `(n_merges: int) -> int` | Run `n_merges` training steps. Returns actual number of merges performed. |
| `save` | `(file_prefix: str) -> None` | Save the trained model to `<file_prefix>.tinymodel`. |

**Properties**

| Property | Type | Description |
|----------|------|-------------|
| `merges` | `list[tuple[int, int]]` | The learned merge pairs. |
| `n_merges` | `int` | Number of merges learned so far. |

**Example**

```python
trainer = SimpleTrainer("hello world hello")

# Manual step-by-step
pair, rank, freq = trainer.step()  # ((104, 101), 256, 2)

# Or use train()
n = trainer.train(500)
print(f"Vocabulary size: {trainer.n_merges + 256}")
trainer.save("my-model")
```

---

## Model I/O

### `BPEParam`

Dataclass holding BPE model parameters.

```python
@dataclass
class BPEParam:
    bytes_maps: list[int] | None   # 256-element byte remapping, or None
    merges: list[tuple[int, int]]  # Merge pairs
```

### `load_bpe_model(model_file: str) -> BPEParam`

Load model parameters from a `.tinymodel` file. Supports versioned and
legacy (unversioned) formats.

### `save_bpe_model(file_prefix: str, merges, bytes_maps=None) -> None`

Save merges and optional byte remapping to `<file_prefix>.tinymodel`.

### `load_bpe_vocab(vocab_file: str) -> dict[int, bytes]`

Load a vocabulary file (tiktoken-compatible format).

### `save_bpe_vocab(file_prefix: str, vocab: dict[int, bytes]) -> None`

Save vocabulary to `<file_prefix>.vocab` in base64-encoded format.

---

## Tiktoken Compatibility

### `get_from_tiktoken(mergeable_ranks: dict[bytes, int]) -> BPEParam`

Convert tiktoken's `enc._mergeable_ranks` into TinyBPE `BPEParam`.
Automatically detects whether byte remapping is needed.

```python
import tiktoken
from tinybpe import get_from_tiktoken, Tokenizer

enc = tiktoken.get_encoding("cl100k_base")
param = get_from_tiktoken(enc._mergeable_ranks)
tokenizer = Tokenizer(param, pat_str, special_tokens=enc._special_tokens)
```

### `save_from_tiktoken(file_prefix: str, mergeable_ranks: dict[bytes, int]) -> None`

Convert and save a tiktoken model as a `.tinymodel` file.

---

## C Extension (`tinybpe.bpe`)

The low-level C extension module. Typically accessed through the Python
wrappers, but can be used directly for byte-level operations.

### `bpe.Trainer`

```python
trainer = bpe.Trainer(list_bytes: list[bytes | bytearray])
trainer.step()                                    # -> (pair, rank, freq) | None
trainer.load_merges(merges: list[tuple[int,int]])  # continue training
trainer.merges                                    # -> list[tuple[int,int]]
trainer.n_merges                                  # -> int
```

### `bpe.Tokenizer`

```python
tokenizer = bpe.Tokenizer(merges, special_tokens=None)
tokenizer.encode(text_bytes: bytes)               # -> list[int]
tokenizer.decode(ids: list[int])                  # -> bytes
tokenizer.cache_decode(token_id: int)             # -> bytes | None
tokenizer.cache_clean()                           # -> None
tokenizer.merges                                  # -> list[tuple[int,int]]
tokenizer.vocab                                   # -> dict[int, bytes]
tokenizer.n_vocab                                 # -> int
```

### `bpe.BytesRemap`

A callable object that remaps every byte through a 256-element lookup table.

```python
mapper = bpe.BytesRemap(list(range(256)))
result = mapper(b"hello")  # bytes with each byte remapped
```
