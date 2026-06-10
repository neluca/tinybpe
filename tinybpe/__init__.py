"""TinyBPE — an ultra-fast, lightweight CPython BPE tokenizer and trainer.

Provides:
  - :class:`Tokenizer` / :class:`CommonTokenizer` — encode/decode with regex pre-tokenization
  - :class:`SimpleTrainer` — train BPE models from text corpora
  - :func:`load_bpe_model` / :func:`save_bpe_model` — model file I/O
  - :func:`get_from_tiktoken` / :func:`save_from_tiktoken` — tiktoken compatibility
"""

__all__ = [
    "__version__",
    "BPEParam",
    "CommonTokenizer",
    "SimpleTrainer",
    "Tokenizer",
    "get_from_tiktoken",
    "load_bpe_model",
    "load_bpe_vocab",
    "save_bpe_model",
    "save_bpe_vocab",
    "save_from_tiktoken",
]

from tinybpe._model_io import BPEParam as BPEParam
from tinybpe._model_io import load_bpe_model as load_bpe_model
from tinybpe._model_io import load_bpe_vocab as load_bpe_vocab
from tinybpe._model_io import save_bpe_model as save_bpe_model
from tinybpe._model_io import save_bpe_vocab as save_bpe_vocab
from tinybpe._tiktoken import get_from_tiktoken as get_from_tiktoken
from tinybpe._tiktoken import save_from_tiktoken as save_from_tiktoken
from tinybpe._version import __version__ as __version__
from tinybpe.core import CommonTokenizer as CommonTokenizer
from tinybpe.core import Tokenizer as Tokenizer
from tinybpe.simple import Trainer as SimpleTrainer
