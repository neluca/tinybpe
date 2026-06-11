"""TinyBPE — an ultra-fast, lightweight CPython BPE tokenizer and trainer.

Provides:

- :class:`Tokenizer` — encode/decode with regex pre-tokenization,
  special token handling, byte remapping, and streaming decode.
- :class:`Trainer` — train BPE models from text corpora.
- :func:`load_model` / :func:`save_model` — ``.tbm`` model file I/O.
- :func:`load_vocab` / :func:`save_vocab` — ``.vocab`` vocabulary file I/O.

Examples
--------
Training::

    >>> from tinybpe import Trainer
    >>> trainer = Trainer("hello world " * 100)
    >>> trainer.train(100)
    100
    >>> trainer.save("my_model")  # → my_model.tbm

Encoding / decoding::

    >>> from tinybpe import Tokenizer
    >>> tok = Tokenizer.from_file("my_model.tbm")
    >>> ids = tok.encode("hello world")
    >>> text = tok.decode(ids)
    >>> assert text == "hello world"

Streaming decode::

    >>> parts = []
    >>> decoder = tok.stream_decode(lambda s: parts.append(s))
    >>> for tid in ids:
    ...     decoder(tid)
    >>> assert "".join(parts) == "hello world"
"""

__all__ = [
    "__version__",
    "Tokenizer",
    "Trainer",
    "load_model",
    "save_model",
    "load_vocab",
    "save_vocab",
]

from tinybpe._model_io import load_model as load_model
from tinybpe._model_io import load_vocab as load_vocab
from tinybpe._model_io import save_model as save_model
from tinybpe._model_io import save_vocab as save_vocab
from tinybpe._version import __version__ as __version__
from tinybpe.tokenizer import Tokenizer as Tokenizer
from tinybpe.trainer import Trainer as Trainer
