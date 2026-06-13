# TinyBPE

[![PyPI version](https://img.shields.io/pypi/v/tinybpe)](https://pypi.org/project/tinybpe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/tinybpe/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

**An ultra-fast, lightweight BPE tokenizer and trainer with a pure-C core.**

Ever wished you could load a GPT-4 compatible tokenizer in **one line** without network calls? TinyBPE ships 8 pre-built ByteLevel BPE models directly in the package. The CPython C core runs BPE encoding/decoding at native speed — typically **10-50× faster** than pure-Python implementations while depending only on `regex`.

## Why TinyBPE?

| Feature | TinyBPE | tiktoken | HuggingFace tokenizers |
|---|---|---|---|
| **Core engine** | Pure C (CPython) | Pure Rust (PyO3) | Pure Rust (PyO3) |
| **Dependencies** | `regex` only | `tiktoken` + Rust toolchain | `tokenizers` + Rust toolchain |
| **Built-in models** | 8 models ship in package | Downloads on first use | Downloads on first use |
| **Offline ready** | ✅ Fully offline | ❌ Requires download | ❌ Requires download |
| **Model format** | Human-readable `.tbm` text | Binary blob | JSON / binary |
| **One-liner load** | `Tokenizer.from_pretrained("cl100k_base")` | `tiktoken.get_encoding("cl100k_base")` | `AutoTokenizer.from_pretrained(...)` |
| **Train new models** | ✅ Pure-C trainer | ❌ | ✅ (requires Rust build) |
| **Streaming decode** | ✅ UTF-8 boundary caching | ❌ | ❌ |
| **Portable C core** | ✅ Embeddable | ❌ | ❌ |
| **Install size** | ~3 MB compressed | ~2 MB + cached models | ~4 MB + cached models |

## Installation

```bash
pip install tinybpe
```

Optional extras:

```bash
pip install tinybpe[dev]       # Development tools (pytest, ruff, mypy)
pip install tinybpe[tiktoken]  # For tiktoken comparison testing
pip install tinybpe[hf]        # For HuggingFace model conversion
pip install tinybpe[all]       # Everything
```

## Quick Start

### One-Line Model Loading

```python
from tinybpe import Tokenizer

# Load any built-in model in one line — no network, no download
tok = Tokenizer.from_pretrained("cl100k_base")

ids = tok.encode("hello world")
tok.decode(ids)  # → 'hello world'
```

### List Available Models

```python
import tinybpe

tinybpe.list_models()
# ['cl100k_base', 'deepseek-llm', 'minicpm', 'o200k_base',
#  'p50k_base', 'phi2', 'qwen25', 'r50k_base']
```

### Built-in Model Catalog

| Model | LLM Compatibility | Vocab Size |
|---|---|---|
| `cl100k_base` | GPT-4, GPT-3.5-turbo, text-embedding-ada-002 | 100,256 |
| `o200k_base` | GPT-4o, GPT-4o-mini, GPT-5 | 199,998 |
| `p50k_base` | GPT-3 (davinci, curie, babbage, ada) | 50,280 |
| `r50k_base` | GPT-2 | 50,256 |
| `qwen25` | Qwen 2.5 (0.5B–72B) | 151,643 |
| `phi2` | Microsoft Phi-2 | 50,257 |
| `deepseek-llm` | DeepSeek V2 (7B-Chat) | 100,013 |
| `minicpm5` | MiniCPM5-1B (ByteLevel BPE) | 130,050 |

### Training

```python
from tinybpe import Trainer

trainer = Trainer("hello world " * 500)
trainer.train(100)          # learn 100 merges
trainer.save("my_model")    # → my_model.tbm
```

### Streaming Decode

```python
parts = []
decoder = tok.stream_decode(lambda s: parts.append(s))
for token_id in ids:
    decoder(tid)
assert "".join(parts) == "hello world"
```

### With Regex Pre-tokenization

```python
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
    @classmethod
    def from_pretrained(cls, name: str) -> Tokenizer

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

### Model Discovery

```python
def list_models() -> list[str]
```

### File I/O

```python
def load_model(path: str) -> tuple[list[tuple[int, int]], list[int] | None]
def save_model(path: str, merges, bytes_maps=None) -> None
def load_vocab(path: str) -> dict[int, bytes]
def save_vocab(path: str, vocab: dict[int, bytes]) -> None
```

## Model Format

`.tbm` (TinyBPE Model) is a human-readable text file:

```
TinyBPE Model v1
0               # 0 = no remap, 256 = has remap
104 101         # merge pairs, one per line
256 108
...
```

See [`docs/file-formats.md`](docs/file-formats.md) for the full specification.

## Conversion Scripts

Convert existing tokenizers to TinyBPE format:

```bash
# TikToken
python scripts/convert_tiktoken.py cl100k_base -o models/cl100k_base.tbm

# HuggingFace
python scripts/convert_hf_tokenizer.py tokenizer.json -o output.tbm
python scripts/convert_hf_tokenizer.py Qwen/Qwen2.5-0.5B -o models/qwen25.tbm

```

See [`scripts/README.md`](scripts/README.md) for details.

## Performance

The C core uses an AVL tree for O(log n) pair lookup during training and greedy lowest-rank-first merging during encoding. Typical throughput on a modern CPU:

| Operation | Tokens/sec |
|---|---|
| Training (C core) | ~5-10M chars/sec |
| Encoding (C core) | ~2-5M tokens/sec |
| Decoding (C core) | ~10-20M tokens/sec |

Run benchmarks locally:

```bash
python benchmarks/bench_train.py
python benchmarks/bench_encode.py
python benchmarks/bench_decode.py
```

## Development

```bash
git clone https://github.com/neluca/tinybpe.git
cd tinybpe
pip install -e ".[dev]"
make test && make lint && make typecheck
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full development setup and PR guidelines.

## License

MIT — see [LICENSE](LICENSE).
