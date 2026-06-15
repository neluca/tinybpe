# Changelog

## [Unreleased]

### Added

- **`count_tokens()`**: new convenience method on `Tokenizer` for counting tokens without the ergonomic overhead of `len(encode(...))`
- **`get_model_info()`**: promoted to public API — returns vocab size, family, description, regex pattern, and special token metadata for any built-in model

### Fixed

- **Version string**: `__version__` corrected from `1.0.0` to `1.1.0` (was missed in the v1.1.0 release)
- **Special tokens for qwen35**: `from_pretrained("qwen35")` now correctly applies special tokens (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`). Models whose special token IDs overlap with byte or merge IDs (deepseek-v4, llama4, minicpm5) now emit a clear warning explaining why special tokens cannot be applied
- **Streaming decode performance**: byte-remap models (cl100k_base, o200k_base, p50k_base, r50k_base) now use a cached O(1) vocab lookup instead of a full batch decode per token, making streaming decode ~100× faster for GPT-family models

### Changed

- **`encode_ordinary` docs**: improved docstring to clearly explain the difference from `encode()` and the behaviour with special tokens

## [1.1.0] — 2026-06-13

### Added

- **Model registry with `from_pretrained`**: `Tokenizer.from_pretrained("cl100k_base")` loads any built-in model in one line
- **`list_models()`**: programmatic model discovery
- **JSON-based model config** (`tinybpe/models/models.json`): add models without code changes
- **8 built-in ByteLevel BPE models**: GPT-4, GPT-4o, GPT-3, GPT-2, Qwen3.5, DeepSeek-V4, Llama 4 Scout, MiniCPM5-1B
- **Special token support**: TikToken models now include FIM tokens, end-of-text markers, etc.
- **Chinese README** (`README_zh.md`)

### Changed

- **Models moved inside package**: `tinybpe/models/` so `pip install` includes them in the wheel
- **`convert_hf_tokenizer.py`**: supports both list-format and string-format merges
- **Upgraded models**: Qwen2.5 → Qwen3.5, DeepSeek V2 → DeepSeek-V4

### Fixed

- Special token regex ordering (longer tokens now match before shorter prefixes)
- `_find_package_file` type error (mypy strict compliance)
- No-op `test_empty_text` replaced with proper assertions
- Dead code and duplicate regex patterns removed
- `encode_ordinary` docstring corrected
- Author name updated to Romani Isa

### Developer Experience

- Makefile with `install`, `test`, `lint`, `format`, `typecheck`, `clean` targets
- `.pre-commit-config.yaml` with ruff + mypy hooks
- Optional dependencies: `[dev]`, `[tiktoken]`, `[hf]`, `[all]`
- CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md
- Issue and PR templates
- Enhanced CI sdist verification (tests `from_pretrained` after install)

## [1.0.0] — 2026-06-12

### Added

- Complete redesign: clean, lightweight Python API
- New `.tbm` (TinyBPE Model) file format — shorter, cleaner extension
- Single `Tokenizer` class with optional byte remapping
- `Tokenizer.from_file()` class method for convenient model loading
- `stream_decode_reset()` for explicit streaming decode cache reset
- `models/` directory for pre-built LLM tokenizer models
- `scripts/` directory with conversion tools (tiktoken, HuggingFace)
- Improved error handling in C extension (safe dealloc on init failure)
- Better default regex pattern that handles multi-line text (`(?s)^.*$`)
- Comprehensive test suite: 60 tests covering C extension, tokenizer, trainer, edge cases, and fuzz

### Changed

- File extension: `.tinymodel` → `.tbm`
- Merged `CommonTokenizer` and `Tokenizer` into single `Tokenizer` class
- Renamed `SimpleTrainer` → `Trainer`
- Removed `_tiktoken.py`, `_utils.py` from the package (conversion scripts live in `scripts/`)
- Simplified public API: 6 exports (`Tokenizer`, `Trainer`, `load_model`, `save_model`, `load_vocab`, `save_vocab`)
- Improved C code comments and organization

### Removed

- `BPEParam` dataclass (internal implementation detail)
- `get_from_tiktoken()`, `save_from_tiktoken()` (use `scripts/convert_tiktoken.py`)
- `CommonTokenizer` (merged into `Tokenizer`)

### v0.x

See the [v0-legacy](https://github.com/neluca/tinybpe/tree/v0-legacy) branch for the v0 codebase.
