"""Built-in model registry.

Maps model names to metadata including file path, vocabulary size,
regex pre-tokenization pattern, and special tokens.

All models ship with the package — no network required.
"""

from __future__ import annotations

from typing import TypedDict


class ModelInfo(TypedDict):
    """Metadata for a built-in BPE model."""

    name: str
    """Short identifier used with :meth:`~tinybpe.Tokenizer.from_pretrained`."""

    path: str
    """Package-relative path to the ``.tbm`` file."""

    vocab_size: int
    """Total vocabulary size (base bytes + merges + special tokens)."""

    description: str
    """Human-readable description including LLM family."""

    family: str
    """LLM family name for grouping."""

    pat_str: str | None
    """Regex pattern for pre-tokenization.  ``None`` means no splitting."""

    special_tokens: dict[str, int] | None
    """Special token strings mapped to their IDs.  ``None`` if none."""

    has_byte_remap: bool
    """Whether the model uses byte remapping (tikToken ByteLevel encoding)."""


# ---------------------------------------------------------------------------
# Regex patterns for ByteLevel BPE pre-tokenization
# ---------------------------------------------------------------------------

# GPT-2 ByteLevel pre-tokenization pattern.
# Used by all TikToken and HuggingFace ByteLevel BPE models.
_PAT_BYTELEVEL = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# No-split pattern (used by MiniCPM and DeepSeek)
_PAT_NONE = None

# ---------------------------------------------------------------------------
# Special token sets
# ---------------------------------------------------------------------------

# GPT-4 / GPT-3.5 special tokens
_SPECIAL_CL100K: dict[str, int] = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}

# o200k special tokens
_SPECIAL_O200K: dict[str, int] = {
    "<|endoftext|>": 199999,
    "<|fim_prefix|>": 200000,
    "<|fim_middle|>": 200001,
    "<|fim_suffix|>": 200002,
    "<|endofprompt|>": 200018,
}

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, ModelInfo] = {
    # ---- TikToken (OpenAI) ----
    "cl100k_base": {
        "name": "cl100k_base",
        "path": "models/cl100k_base.tbm",
        "vocab_size": 100277,
        "description": "GPT-4 / GPT-3.5-turbo / text-embedding-ada-002",
        "family": "GPT-4",
        "pat_str": _PAT_BYTELEVEL,
        "special_tokens": _SPECIAL_CL100K,
        "has_byte_remap": True,
    },
    "o200k_base": {
        "name": "o200k_base",
        "path": "models/o200k_base.tbm",
        "vocab_size": 200019,
        "description": "GPT-4o / GPT-4o-mini / GPT-5",
        "family": "GPT-4o",
        "pat_str": _PAT_BYTELEVEL,
        "special_tokens": _SPECIAL_O200K,
        "has_byte_remap": True,
    },
    "p50k_base": {
        "name": "p50k_base",
        "path": "models/p50k_base.tbm",
        "vocab_size": 50281,
        "description": "GPT-3 (davinci, curie, babbage, ada)",
        "family": "GPT-3",
        "pat_str": _PAT_BYTELEVEL,
        "special_tokens": {"<|endoftext|>": 50256},
        "has_byte_remap": True,
    },
    "r50k_base": {
        "name": "r50k_base",
        "path": "models/r50k_base.tbm",
        "vocab_size": 50257,
        "description": "GPT-2",
        "family": "GPT-2",
        "pat_str": _PAT_BYTELEVEL,
        "special_tokens": {"<|endoftext|>": 50256},
        "has_byte_remap": True,
    },
    # ---- HuggingFace ByteLevel BPE ----
    "qwen35": {
        "name": "qwen35",
        "path": "models/qwen35.tbm",
        "vocab_size": 247843,
        "description": "Qwen3.5 (0.8B-35B)",
        "family": "Qwen",
        "pat_str": _PAT_BYTELEVEL,
        "special_tokens": None,
        "has_byte_remap": False,
    },
    "phi2": {
        "name": "phi2",
        "path": "models/phi2.tbm",
        "vocab_size": 51200,
        "description": "Microsoft Phi-2",
        "family": "Phi-2",
        "pat_str": _PAT_BYTELEVEL,
        "special_tokens": None,
        "has_byte_remap": True,
    },
    "deepseek-llm": {
        "name": "deepseek-llm",
        "path": "models/deepseek-llm.tbm",
        "vocab_size": 100000,
        "description": "DeepSeek V2 (7B-Chat)",
        "family": "DeepSeek",
        "pat_str": None,
        "special_tokens": None,
        "has_byte_remap": True,
    },
    # ---- ByteLevel BPE (no regex) ----
    "minicpm5": {
        "name": "minicpm5",
        "path": "models/minicpm5.tbm",
        "vocab_size": 130050,
        "description": "MiniCPM5-1B (ByteLevel BPE)",
        "family": "MiniCPM",
        "pat_str": None,
        "special_tokens": None,
        "has_byte_remap": False,
    },
}


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
    ['cl100k_base', 'deepseek-llm', 'minicpm5', 'o200k_base', 'p50k_base', 'phi2', 'qwen35', 'r50k_base']
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
