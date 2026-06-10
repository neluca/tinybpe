# Contributing to TinyBPE

Thank you for your interest in contributing! TinyBPE is a CPython extension implementing a fast, lightweight BPE tokenizer and trainer.

## Development Setup

### Prerequisites

- Python 3.9+
- A C compiler (GCC on Linux, Clang on macOS, MSVC on Windows)
- Git

### Setting up the development environment

```bash
# Clone the repository
git clone https://github.com/neluca/tinybpe.git
cd tinybpe

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install development dependencies
pip install -r requirements_dev.txt

# Build the C extension in-place
pip install -e .

# Run tests
python -m pytest
```

### Building the C extension manually

```bash
# Build the extension module (places .pyd/.so in the source tree)
python setup.py build_ext --inplace

# Or use pip for an editable install
pip install -e .
```

## Coding Conventions

### C code (`src/`)
- **Standard**: C99
- **Style**: Consistent with existing code — 4-space indentation, snake_case for function and variable names
- **Naming**: Use descriptive names. Struct fields use `snake_case`. Functions are prefixed with `bpe_`.
- **Memory**: Use `bpe_malloc` / `bpe_free` wrappers which integrate with Python's memory manager. Always check for NULL returns.
- **Error handling**: Set Python exceptions via `PyErr_SetString` / `PyErr_NoMemory` and return NULL or -1.

### Python code (`tinybpe/`)
- **Standard**: Python 3.9+ compatible
- **Style**: Follow [PEP 8](https://peps.python.org/pep-0008/) — enforced by ruff
- **Type hints**: All public methods must have type annotations
- **Docstrings**: NumPy-style docstrings required for all public classes and methods

### Pre-commit hooks

```bash
# Install pre-commit hooks (run once)
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

## Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=tinybpe --cov-report=term

# Run a specific test file
python -m pytest tests/test_tinybpe.py
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, following the coding conventions above.
3. Add or update tests as needed.
4. Run `pre-commit run --all-files` to ensure style compliance.
5. Run `python -m pytest` to verify all tests pass.
6. Update `CHANGELOG.md` under the `[Unreleased]` section.
7. Submit a pull request with a clear description of the changes.

## Project Structure

```
tinybpe/
├── src/                  # C extension source
│   ├── _tree_core.h/c    # AVL tree (self-balancing binary search tree)
│   ├── bpe_common.h/c    # Common types (pairs, pieces) and utilities
│   ├── bpe_tokenizer.h/c # Encoder/decoder implementation
│   ├── bpe_trainer.h/c   # BPE training algorithm
│   └── bpe_module.c      # CPython type definitions and method wrappers
├── tinybpe/              # Python package
│   ├── __init__.py       # Public API exports
│   ├── _version.py       # Single source of truth for version
│   ├── _model_io.py      # Model file save/load
│   ├── _tiktoken.py      # tiktoken model conversion
│   ├── core.py           # CommonTokenizer and Tokenizer wrappers
│   ├── simple.py         # SimpleTrainer wrapper
│   └── bpe.pyi           # Type stubs for the C extension
├── tests/                # Test suite
├── examples/             # Example scripts
├── benchmarks/           # Performance benchmarks
└── .github/workflows/    # CI/CD pipelines
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
