"""Built-in model registry.

Loads model metadata from ``models/models.json``.  Each model entry
specifies its ``.tbm`` file path, vocabulary size, regex pre-tokenization
pattern, special tokens, and byte-remap status.

All models ship with the package — no network required.

Adding a new model:

1. Place the ``.tbm`` file in ``models/``.
2. Add a JSON entry to ``models/models.json``.
3. Done — :func:`list_models` and :meth:`~tinybpe.Tokenizer.from_pretrained`
   pick it up automatically.
"""

from __future__ import annotations

import json
import os
from typing import TypedDict


class ModelInfo(TypedDict):
    """Metadata for a built-in BPE model."""

    name: str
    path: str
    vocab_size: int
    description: str
    family: str
    pat_str: str | None
    special_tokens: dict[str, int] | None
    has_byte_remap: bool


# ---------------------------------------------------------------------------
# Load registry from JSON
# ---------------------------------------------------------------------------

def _load_registry() -> tuple[dict[str, ModelInfo], dict[str, str | None]]:
    """Load model registry and pattern definitions from ``models/models.json``.

    Returns
    -------
    tuple
        ``(registry, patterns)`` where *registry* maps model names to
        ``ModelInfo``, and *patterns* maps pattern names to regex strings
        (or ``None`` for no-split).
    """
    # Resolve the JSON path.  Models live inside the package: tinybpe/models/
    json_path = os.path.join(os.path.dirname(__file__), "models", "models.json")
    # Also try importlib.resources for wheel installs
    if not os.path.isfile(json_path):
        try:
            from importlib.resources import files as _files
            from pathlib import Path

            json_path = str(Path(str(_files("tinybpe"))) / "models" / "models.json")
        except Exception:
            pass

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Resolve pattern references
    pattern_map: dict[str, str | None] = data.get("patterns", {})

    registry: dict[str, ModelInfo] = {}
    for entry in data.get("models", []):
        pat_ref: str = entry.get("pattern", "none")
        pat_str = pattern_map.get(pat_ref)

        # Special tokens are only valid for byte-remap models (tiktoken)
        # where token IDs are preserved.  For ID-remapped models, the
        # remapped IDs differ from the originals.
        raw_special: dict[str, int] | None = entry.get("special_tokens")
        special_tokens: dict[str, int] | None
        if entry.get("has_byte_remap", False) and raw_special:
            special_tokens = raw_special
        elif raw_special:
            # ID-remapped: store for reference but don't apply
            special_tokens = None
        else:
            special_tokens = None

        info: ModelInfo = {
            "name": entry["name"],
            "path": entry["path"],
            "vocab_size": entry["vocab_size"],
            "description": entry["description"],
            "family": entry["family"],
            "pat_str": pat_str,
            "special_tokens": special_tokens,
            "has_byte_remap": entry.get("has_byte_remap", False),
        }
        registry[info["name"]] = info

    return registry, pattern_map


_MODEL_REGISTRY, _PATTERNS = _load_registry()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_models() -> list[str]:
    """Return a sorted list of all built-in model names.

    These names can be passed to :meth:`tinybpe.Tokenizer.from_pretrained`.

    Returns
    -------
    list[str]
        Alphabetically sorted model names.

    Examples
    --------
    >>> import tinybpe
    >>> tinybpe.list_models()
    ['cl100k_base', 'deepseek-v4', 'llama4', 'minicpm5', 'o200k_base', 'p50k_base', 'qwen35', 'r50k_base']
    """
    return sorted(_MODEL_REGISTRY.keys())


def get_model_info(name: str) -> ModelInfo:
    """Get metadata for a built-in model.

    Parameters
    ----------
    name : str
        Model name (see :func:`list_models`).

    Returns
    -------
    ModelInfo
        Model metadata including path, vocab size, regex pattern, and
        special tokens.

    Raises
    ------
    ValueError
        If the model name is not found.

    Examples
    --------
    >>> from tinybpe._registry import get_model_info
    >>> info = get_model_info("cl100k_base")
    >>> info["vocab_size"]
    100277
    """
    if name not in _MODEL_REGISTRY:
        available = list_models()
        raise ValueError(f"Unknown model {name!r}. Available models: {available}")
    return _MODEL_REGISTRY[name]
