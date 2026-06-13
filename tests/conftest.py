"""Shared test fixtures for TinyBPE test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def models_dir() -> str:
    """Return the absolute path to the ``models/`` directory.

    Works in both editable installs and wheel installs.
    """
    # Try common locations
    candidates = [
        # Editable install: models/ next to tests/
        os.path.join(os.path.dirname(__file__), "..", "models"),
        # Absolute path relative to cwd
        os.path.join(os.getcwd(), "models"),
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("models/ directory not found")


@pytest.fixture(scope="session")
def models_dir_path(models_dir: str) -> Path:
    """Return the models directory as a ``pathlib.Path``."""
    return Path(models_dir)
