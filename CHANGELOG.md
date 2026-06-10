# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-06-11

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
- Codecov coverage badge in README
- Edge case and fuzz tests for robust coverage
- Benchmarks for encode, decode, and training
- Chinese README (`README_zh.md`)
- `MANIFEST.in` for proper sdist packaging
- `.gitattributes` for consistent LF line endings

### Changed
- Renamed `Tokenizer.size` to `Tokenizer.n_vocab` for naming consistency
- Renamed `Trainer.merges_size` to `Trainer.n_merges` for naming consistency
- Split `_utils.py` into `_model_io.py` and `_tiktoken.py`
- Improved `save_bpe_vocab` to write tab-separated, machine-parseable format
- Use `X | Y` type annotation syntax (PEP 604)
- Updated CI from flake8 to ruff
- Removed `build_setup.py` in favor of standard setuptools build
- Updated `macos-13` to `macos-latest` in CI workflows

### Fixed
- Integer truncation bug in pair validation (`(int)` cast on unsigned long)
- Missing NULL checks after `bpe_encode`, `bpe_decode`, `bpe_malloc` calls
- Duplicate `bpe_utf8_head_check` function removed
- `bpe_utf8_length_from_head` returning wrong value for continuation bytes
- Circular import when importing `tinybpe` during build
- `bytes_remap_init` not setting Python error before returning -1
- Bare `assert` replaced with explicit `ValueError` in `_tiktoken.py`
- `AVL_MAX`/`AVL_ABS` macros renamed to avoid stdlib shadowing
- Removed `cp313t-*` free-threaded builds (not supported by cibuildwheel 4.x)
- CI workflow fixes for build isolation and sdist verification

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
