# TinyBPE v1.1.0 Release Notes

**An ultra-fast, lightweight ByteLevel BPE tokenizer with a pure-C core.**

---

## Highlights

- **8 built-in ByteLevel BPE models** — GPT-4, GPT-4o, GPT-3, GPT-2, Qwen3.5, DeepSeek-V4, Llama 4 Scout, MiniCPM5-1B
- **One-line model loading** — `Tokenizer.from_pretrained("cl100k_base")`, no network required
- **JSON-based model registry** — add new models by editing `models.json`, no code changes
- **96% test coverage** — 203 tests, mypy strict, ruff clean
- **Professional packaging** — PyPI-ready with pre-built wheels, CONTRIBUTING/SECURITY/CODE_OF_CONDUCT

---

## New Features

### Model Registry & One-Line Loading (`#41e7685`)
```python
from tinybpe import Tokenizer, list_models

list_models()
# ['cl100k_base', 'deepseek-v4', 'llama4', 'minicpm5', 'o200k_base', 'p50k_base', 'qwen35', 'r50k_base']

tok = Tokenizer.from_pretrained("cl100k_base")
ids = tok.encode("hello world")
```

### JSON-Based Model Configuration (`#3c44355`)
All model metadata (name, vocab size, regex pattern, special tokens) lives in `tinybpe/models/models.json`. Adding a new model:
1. Drop the `.tbm` file into `tinybpe/models/`
2. Add a JSON entry to `models.json`

### Special Token Support
TikToken models now support special tokens out of the box:
```python
tok = Tokenizer.from_pretrained("cl100k_base")
ids = tok.encode("<|endoftext|> hello <|fim_prefix|>")
# → [100257, 24748, 220, 100258]
```

---

## Model Updates

| Action | Model | Vocab | Notes |
|---|---|---|---|
| **Upgraded** | Qwen2.5 → Qwen3.5 | 151K → **248K** | ByteLevel BPE with ID remapping |
| **Upgraded** | DeepSeek V2 → DeepSeek-V4 | 100K → **128K** | ByteLevel BPE with ID remapping |
| **Added** | Llama 4 Scout (17B) | **440K** | ByteLevel BPE with ID remapping |
| **Added** | MiniCPM5-1B | **130K** | ByteLevel BPE with ID remapping |
| **Removed** | MiniCPM-2B | 123K | SentencePiece BPE (no longer supported) |
| **Removed** | Phi-2 | 50K | Superseded by newer models |

---

## Bug Fixes

- **Special token regex ordering** — tokens sorted by length descending so `"<ab>"` matches before `"<a>"` (`#866502b`)
- **mypy type errors** — `_find_package_file` type inconsistency fixed
- **No-op test** — `test_empty_text` replaced with actual assertions
- **Dead code** — removed unused variable assignment in `convert_minicpm.py`
- **UTF-8 continuation byte check** — `convert_hf_tokenizer.py` now validates bytes 0x80-0xBF
- **Docstring mismatches** — fixed `encode_ordinary` description and vocab format docs
- **Duplicate regex patterns** — `_PAT_GPT2` and `_PAT_HF_BYTELEVEL` consolidated into `_PAT_BYTELEVEL`
- **Author name** — updated to Romani Isa

---

## Developer Experience

- **Makefile** — `make install`, `make test`, `make lint`, `make format`, `make typecheck`, `make clean`
- **pre-commit hooks** — ruff + ruff-format + mypy
- **Optional dependencies** — `pip install tinybpe[dev|tiktoken|hf|all]`
- **Community files** — CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md, issue templates, PR template
- **README_zh.md** — Chinese translation of the full README

---

## Technical Details

- **Ruff**: zero lint errors, full formatting compliance
- **mypy**: strict mode, zero errors across 7 source files
- **Test suite**: 203 tests, 96% branch coverage
- **Wheel size**: ~6 MB (includes all 8 models + models.json)
- **Python**: 3.9–3.13 supported, 3.14 classified
- **Platforms**: Linux (manylinux2014, x86_64 + aarch64), macOS, Windows

---

## What's Changed (since v1.0.0)

- 28 commits
- 29 files changed (new), 17 files modified
- ~380K lines of model data added
- Author changed to Romani Isa

---

## Installation

```bash
pip install tinybpe
```

## Quick Start

```python
from tinybpe import Tokenizer

# Load any built-in model — no network
tok = Tokenizer.from_pretrained("cl100k_base")
ids = tok.encode("hello world")
print(tok.decode(ids))  # → 'hello world'
```

---

**Full Changelog**: https://github.com/neluca/tinybpe/commits/v1.1.0
