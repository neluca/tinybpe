[English] | [中文](README_zh.md)

# 🚀 TinyBPE

[![build](https://github.com/neluca/tinybpe/workflows/build/badge.svg)](https://github.com/neluca/tinybpe/actions/workflows/python-package.yml)
[![wheels](https://github.com/neluca/tinybpe/workflows/wheels/badge.svg)](https://github.com/neluca/tinybpe/actions/workflows/wheels.yml)
[![lint](https://github.com/neluca/tinybpe/workflows/lint/badge.svg)](https://github.com/neluca/tinybpe/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/neluca/tinybpe/branch/main/graph/badge.svg)](https://codecov.io/gh/neluca/tinybpe)
[![PyPI version](https://img.shields.io/pypi/v/tinybpe)](https://pypi.org/project/tinybpe/)
[![Python versions](https://img.shields.io/pypi/pyversions/tinybpe)](https://pypi.org/project/tinybpe/)
[![License](https://img.shields.io/github/license/neluca/tinybpe)](https://github.com/neluca/tinybpe/blob/main/LICENSE)

**TinyBPE** is an ultra-fast, lightweight, and clean **language model** tokenizer and BPE model trainer implemented as a **CPython** extension.

## 📦 Installation

```bash
pip install tinybpe
```

Pre-built wheels are available for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x86_64), for Python 3.9–3.13.

## 🌟 Features

- **C core** — Meticulously designed C implementation using AVL-tree indexing for fast pair lookup.
- **Clean Python API** — Simple, elegant interface with type hints.
- **BPE training** — Train from scratch or continue training on imported models.
- **Byte-level tokenizer** — Fast encode/decode with streaming decode support.
- **Regex pre-tokenization** — Split text before encoding using regex patterns.
- **Special tokens** — Support for control tokens like `<|endoftext|>`.
- **TikToken compatibility** — Convert tiktoken model parameters for use with tinybpe.
- **Zero core dependencies** — The C extension has zero dependencies; only `regex` is needed for pre-tokenization.

## ⚡️ Quick Start

### 1. Basic Tokenization

```python
import tiktoken
from tinybpe import Tokenizer, get_from_tiktoken

# Convert a tiktoken model
tik_tokenizer = tiktoken.get_encoding("cl100k_base")
model_param = get_from_tiktoken(tik_tokenizer._mergeable_ranks)
tiny_tokenizer = Tokenizer(model_param)

text = "👋 Hello, this is an example. 你好，这是一个例子。😁"
tik_ids = tik_tokenizer.encode(text)
tiny_ids = tiny_tokenizer.encode(text)
assert tik_ids == tiny_ids  # Identical output
```

### 2. Training a BPE Model

```python
from tinybpe import SimpleTrainer

text = open("corpus.txt", "r", encoding="utf-8").read()
trainer = SimpleTrainer(text)
vocab_size = 1000
for _ in range(vocab_size - 256):
    pair, rank, freq = trainer.step()
    print(f"{pair} -> {rank} ({freq})")

print(f"Vocabulary size: {trainer.n_merges + 256}")
trainer.save("my-model")  # Saves my-model.tinymodel
```

### 3. Loading a Model

```python
from tinybpe import Tokenizer, load_bpe_model

model = load_bpe_model("my-model.tinymodel")
tokenizer = Tokenizer(model)

ids = tokenizer.encode("hello world")
print(ids)                      # [259, 32, 261, 263, 264]
print(tokenizer.decode(ids))    # hello world
print(tokenizer.n_vocab)        # 1000
```

### 4. Streaming Decode

```python
def on_text(text: str):
    print(text, end="")

decode = tokenizer.stream_decode(on_text)
for token_id in ids:
    decode(token_id)  # Prints characters as soon as they're decodable
```

### 5. Convert TikToken Models

```python
import tiktoken
from tinybpe import save_from_tiktoken

enc = tiktoken.get_encoding("cl100k_base")
save_from_tiktoken("cl100k_base", enc._mergeable_ranks)
# Creates cl100k_base.tinymodel
```

**Note:** In commercial settings, be mindful of copyright when converting third-party tokenizer models. Training your own model is recommended.

## 🧪 Development

```bash
git clone https://github.com/neluca/tinybpe.git
cd tinybpe
pip install -r requirements_dev.txt
pip install -e .
python -m pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development setup and guidelines.

## 📊 Benchmarks

Run benchmarks with:
```bash
cd benchmarks
python bench_encode.py
python bench_decode.py
python bench_train.py
```

TinyBPE's C implementation typically achieves **10–100x faster** encoding than pure-Python BPE implementations.

## 🤝 Acknowledgements

- [minbpe](https://github.com/karpathy/minbpe) — Excellent educational resource on BPE algorithm internals.
- [tiktoken](https://github.com/openai/tiktoken) — Reference tokenizer models for validation and compatibility.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
