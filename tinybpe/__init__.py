"""TinyBPE — an ultra-fast, lightweight CPython BPE tokenizer and trainer.

Provides:

- :class:`Tokenizer` — encode/decode with regex pre-tokenization,
  special token handling, byte remapping, and streaming decode.
- :func:`list_models` — list built-in models available via
  :meth:`Tokenizer.from_pretrained`.
- :class:`Trainer` — train BPE models from text corpora.
- :func:`load_model` / :func:`save_model` — ``.tbm`` model file I/O.
- :func:`load_vocab` / :func:`save_vocab` — ``.vocab`` vocabulary file I/O.

Examples
--------
One-line loading a built-in model::

    >>> from tinybpe import Tokenizer
    >>> tok = Tokenizer.from_pretrained("cl100k_base")
    >>> ids = tok.encode("hello world")
    >>> tok.decode(ids)
    'hello world'

Training from scratch::

    >>> from tinybpe import Trainer
    >>> trainer = Trainer("hello world " * 100)
    >>> trainer.train(100)
    100
    >>> trainer.save("my_model")  # → my_model.tbm

Streaming decode::

    >>> parts = []
    >>> decoder = tok.stream_decode(lambda s: parts.append(s))
    >>> for tid in ids:
    ...     decoder(tid)
    >>> assert "".join(parts) == "hello world"

List available models::

    >>> import tinybpe
    >>> tinybpe.list_models()
    ['cl100k_base', 'deepseek-llm', 'minicpm5', 'o200k_base', 'p50k_base', 'phi2', 'qwen35', 'r50k_base']
"""

__all__ = [
    "Tokenizer",
    "Trainer",
    "__version__",
    "list_models",
    "load_model",
    "load_vocab",
    "save_model",
    "save_vocab",
]

from tinybpe._model_io import load_model as load_model
from tinybpe._model_io import load_vocab as load_vocab
from tinybpe._model_io import save_model as save_model
from tinybpe._model_io import save_vocab as save_vocab
from tinybpe._registry import list_models as list_models
from tinybpe._version import __version__ as __version__
from tinybpe.tokenizer import Tokenizer as Tokenizer
from tinybpe.trainer import Trainer as Trainer
