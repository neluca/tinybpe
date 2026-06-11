# Changelog

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
