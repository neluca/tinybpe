# TinyBPE

[![PyPI version](https://img.shields.io/pypi/v/tinybpe)](https://pypi.org/project/tinybpe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An ultra-fast, lightweight BPE tokenizer and trainer with a pure-C core.**

TinyBPE implements the Byte Pair Encoding (BPE) algorithm as a CPython extension. The C core is extremely efficient — AVL-tree-based pair lookup, greedy encoding, and single-allocation vocabulary — while the Python layer adds regex pre-tokenization, special token handling, and streaming decode.

## Features

- **Blazing fast** — C-level AVL tree for O(log n) pair lookup
- **Minimal dependencies** — only `regex`
- **Streaming decode** — token-by-token UTF-8 reassembly
- **TikToken compatible** — byte remapping for GPT tokenizer models
- **Lightweight** — single `.tbm` model file
- **Portable C core** — trainer and tokenizer are pure C, ready for embedded use

## Installation

```bash
pip install tinybpe
```

## Quick Start

### Training

```python
from tinybpe import Trainer

# Train on text
trainer = Trainer("hello world " * 500)
trainer.train(100)          # learn 100 merges
trainer.save("my_model")    # → my_model.tbm
```

### Encoding / Decoding

```python
from tinybpe import Tokenizer

# Load a model
tok = Tokenizer.from_file("my_model.tbm")

# Encode text to token IDs
ids = tok.encode("hello world")
# [265, 267, 108, ...]

# Decode back to text
text = tok.decode(ids)
assert text == "hello world"
```

### Streaming Decode

```python
parts = []
decoder = tok.stream_decode(lambda s: parts.append(s))
for token_id in ids:
    decoder(token_id)
assert "".join(parts) == "hello world"
```

### With Regex Pre-tokenization

```python
import regex as re

PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

tok = Tokenizer.from_file("my_model.tbm", pat_str=PAT)
```

### With Special Tokens

```python
special_tokens = {"<eot>": 1000, "<fim_prefix>": 1001, "<fim_suffix>": 1002}
tok = Tokenizer(merges, special_tokens=special_tokens)
ids = tok.encode("<fim_prefix> hello world <eot>")
```

### With Byte Remapping (TikToken Compat)

```python
from tinybpe import load_model

merges, bytes_maps = load_model("cl100k_base.tbm")
tok = Tokenizer(merges, bytes_maps=bytes_maps)
```

## API Reference

### `Tokenizer`

```python
class Tokenizer:
    def __init__(self, merges, *, bytes_maps=None, pat_str=None, special_tokens=None)
    def encode(self, text: str) -> list[int]
    def encode_ordinary(self, text: str) -> list[int]
    def decode(self, ids: list[int]) -> str
    def stream_decode(self, callback: Callable[[str], None]) -> Callable[[int], None]
    def stream_decode_reset(self) -> None
    def save(self, path: str) -> None
    def save_vocab(self, path: str) -> None

    @classmethod
    def from_file(cls, path: str, *, pat_str=None, special_tokens=None) -> Tokenizer

    @property
    def merges(self) -> list[tuple[int, int]]
    @property
    def vocab(self) -> dict[int, bytes]
    @property
    def n_vocab(self) -> int
```

### `Trainer`

```python
class Trainer(bpe.Trainer):
    def __init__(self, text, *, preprocess=None, callback=None)
    def step(self) -> tuple | None
    def train(self, n: int) -> int
    def save(self, path: str) -> None

    @property
    def merges(self) -> list[tuple[int, int]]
    @property
    def n_merges(self) -> int
```

### File I/O

```python
def load_model(path: str) -> tuple[list[tuple[int, int]], list[int] | None]
def save_model(path: str, merges, bytes_maps=None) -> None
def load_vocab(path: str) -> dict[int, bytes]
def save_vocab(path: str, vocab: dict[int, bytes]) -> None
```

## Model Format

`.tbm` (TinyBPE Model) is a text file:

```
TinyBPE Model v1
0               # 0 = no remap, 256 = has remap
[left] [right]  # merge pairs, one per line
...
```

## Conversion Scripts

Convert existing tokenizers to TinyBPE format:

```bash
# TikToken
python scripts/convert_tiktoken.py cl100k_base -o models/cl100k_base.tbm

# HuggingFace
python scripts/convert_hf_tokenizer.py tokenizer.json -o output.tbm
```

## License

MIT — see [LICENSE](LICENSE).
