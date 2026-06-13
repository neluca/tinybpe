.PHONY: install install-dev test lint format typecheck all clean

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

test:
	pytest --cov=tinybpe --cov-report=term-missing tests/

test-quick:
	pytest tests/ -x -q

lint:
	ruff check tinybpe/ tests/ scripts/

format:
	ruff format tinybpe/ tests/ scripts/

format-check:
	ruff format --check tinybpe/ tests/ scripts/

typecheck:
	mypy tinybpe/ --strict

all: format lint typecheck test

clean:
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
