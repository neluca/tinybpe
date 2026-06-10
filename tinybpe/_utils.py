"""Backward-compatibility re-exports from _model_io and _tiktoken.

Prefer importing directly from ``tinybpe._model_io`` and ``tinybpe._tiktoken``.
"""

from tinybpe._model_io import (
    BPEParam,
    load_bpe_model,
    load_bpe_vocab,
    save_bpe_model,
    save_bpe_vocab,
)
from tinybpe._tiktoken import (
    get_from_tiktoken,
    save_from_tiktoken,
)

__all__ = [
    "BPEParam",
    "load_bpe_model",
    "load_bpe_vocab",
    "save_bpe_model",
    "save_bpe_vocab",
    "get_from_tiktoken",
    "save_from_tiktoken",
]
