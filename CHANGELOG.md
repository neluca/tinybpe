# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `SECURITY.md` with vulnerability reporting process
- `CONTRIBUTING.md` with development setup guide
- `CODE_OF_CONDUCT.md` adopting Contributor Covenant
- `py.typed` marker for PEP 561 compliance
- Pre-commit hooks configuration (ruff, mypy)
- `_version.py` for lightweight version access without C extension import
- `n_vocab` property as consistent alias for vocabulary size
- `n_merges` property as consistent alias for merges count
- `train()` convenience method on Trainer with optional progress callback
- Bounds checking in `bpe_decode` to prevent segfault on invalid IDs
- Overflow guards on memory allocations in C layer
- Model file format versioning (`TinyBPE Model v1`)

### Changed
- Renamed `Tokenizer.size` to `Tokenizer.n_vocab` for naming consistency
- Renamed `Trainer.merges_size` to `Trainer.n_merges` for naming consistency
- Split `_utils.py` into `_model_io.py` and `_tiktoken.py`
- Improved `save_bpe_vocab` to write tab-separated, machine-parseable format
- Added NULL-safety to `bpe_merges_free` and `bpe_vocab_free`
- Optimized list flattening in `encode_ordinary` to avoid O(n²) behavior
- Updated CI from flake8 to ruff
- Removed `build_setup.py` in favor of standard setuptools build

### Fixed
- Integer truncation bug in pair validation (`(int)` cast on unsigned long)
- Missing NULL checks after `bpe_encode`, `bpe_decode`, `bpe_malloc` calls
- Duplicate `bpe_utf8_head_check` function removed
- C struct fields `_1`/`_2` renamed to `left`/`right` for clarity

### Security
- Added bounds checking in `bpe_decode` to prevent out-of-bounds memory access
- Added overflow guards on all size computations before `malloc`

## [0.1.1] - 2025-07-10

### Added
- Initial public release
- CPython BPE tokenizer (encode, decode, streaming decode)
- CPython BPE trainer with support for continuing training
- AVL-tree based merge index for fast pair lookup
- Byte remapping support (compatible with tiktoken models)
- Regex pre-tokenization support
- Special tokens support
- tiktoken model conversion utilities
- Pre-built wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x86_64)
- CI/CD pipelines for build, test, lint, and PyPI release
