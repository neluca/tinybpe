# Contributing to TinyBPE

Thank you for your interest in contributing! TinyBPE is an ultra-fast BPE tokenizer with a pure-C core.

## Development Setup

```bash
git clone https://github.com/neluca/tinybpe.git
cd tinybpe
pip install -e ".[dev]"
```

This installs TinyBPE in editable mode with all development dependencies.

## Running Tests

```bash
pytest --cov=tinybpe --cov-report=term-missing tests/
```

All tests must pass and coverage should stay above 90%.

## Code Style

We use **ruff** for linting and formatting, and **mypy** for type checking:

```bash
# Format
ruff format tinybpe/ tests/ scripts/

# Lint
ruff check tinybpe/ tests/ scripts/

# Type check
mypy tinybpe/ --strict
```

- Line length: 120
- Quote style: double quotes
- Strict mypy checking on the `tinybpe/` package
- Python 3.9+ compatibility required
- An [`.editorconfig`](.editorconfig) is provided for consistent editor settings (indentation, line endings, charset).

## Project Structure

```
tinybpe/           # Python package
  __init__.py      # Public API exports
  tokenizer.py     # Tokenizer class
  trainer.py       # Trainer class
  _model_io.py     # .tbm / .vocab file I/O
  _registry.py     # Built-in model catalog
src/               # C source code
  bpe_module.c     # CPython extension
  bpe_tokenizer.c  # BPE encode/decode engine
  bpe_trainer.c    # BPE training engine
  _tree_core.c     # AVL tree
models/            # Pre-built .tbm model files
scripts/           # Conversion utilities
tests/             # Test suite
```

## Adding a New Built-in Model

1. Convert the tokenizer to `.tbm` format using a conversion script (or write a new one in `scripts/`).
2. Place the `.tbm` file in `models/`.
3. Add an entry to `_MODEL_REGISTRY` in `tinybpe/_registry.py` with the correct metadata (vocab size, regex pattern, special tokens).
4. Add a test in `tests/test_model_registry.py` that loads the model via `Tokenizer.from_pretrained()` and verifies roundtrip correctness.

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Make your changes, including tests.
3. Run `ruff format`, `ruff check`, and `mypy --strict` on changed files.
4. Run the full test suite with `pytest tests/`.
5. Submit a PR against the `main` branch with a clear description.

## Release Process

Releases are automated via GitHub Actions. When a new GitHub Release is created with a version tag (e.g., `v1.1.0`), the CI pipeline builds wheels for all platforms and publishes to PyPI.

Manual release steps:
```bash
# Update version in tinybpe/_version.py
# Update CHANGELOG.md
git tag vX.Y.Z
git push --tags
# Create a GitHub Release from the tag
```

## Code of Conduct

Please see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
