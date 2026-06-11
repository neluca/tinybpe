"""High-level BPE trainer with text encoding and progress callbacks.

Provides the :class:`Trainer` class which wraps the C-level
``bpe.Trainer`` with automatic UTF-8 text encoding and optional
preprocessing.
"""

from __future__ import annotations

from typing import Callable

import tinybpe.bpe as bpe
from tinybpe._model_io import save_model


class Trainer(bpe.Trainer):
    """A simple Byte-Pair-Encoding trainer.

    Extends the C-level ``bpe.Trainer`` with automatic UTF-8 text
    encoding and an optional preprocessing callback.

    Parameters
    ----------
    text : str
        The training text (UTF-8 encoded if no ``preprocess`` is given).
    preprocess : callable or None
        Optional function that takes the raw text and returns a list of
        ``bytes`` or ``bytearray`` chunks.  Use this for regex
        pre-tokenization (splitting text before training).
    callback : callable or None
        Optional progress callback called after each training step.
        Receives ``(step, total, pair, rank, frequency)`` where
        ``total`` may be 0 if the total is unknown.

    Examples
    --------
    Basic training::

        >>> trainer = Trainer("hello world hello")
        >>> trainer.step()
        ((104, 101), 256, 2)

    Training with regex preprocessing::

        >>> import regex as re
        >>> pat = re.compile(r"\\\\w+|\\\\s+")
        >>> def preprocess(text):
        ...     return [ch.encode() for ch in pat.findall(text)]
        >>> trainer = Trainer(text, preprocess=preprocess)
        >>> trainer.train(1000)
        1000

    Training with progress reporting::

        >>> def on_step(step, total, pair, rank, freq):
        ...     print(f"Step {step}: {pair} -> {rank}")
        >>> trainer = Trainer("hello world", callback=on_step)

    Continue training from an existing model::

        >>> trainer = Trainer("new text")
        >>> trainer.load_merges(existing_merges)  # inherit from bpe.Trainer
        >>> trainer.train(50)

    Note: ``save()`` saves only the merge pairs and (if applicable) byte
    remapping.  The regex pattern, preprocess callback, special tokens,
    and training state are NOT preserved.
    """

    def __init__(
        self,
        text: str,
        preprocess: Callable[[str], list[bytes | bytearray]] | None = None,
        *,
        callback: (Callable[[int, int, tuple[int, int], int, int], None] | None) = None,
    ) -> None:
        if preprocess is None:
            pieces: list[bytes | bytearray] = [text.encode("utf-8")]
        else:
            pieces = preprocess(text)

        super().__init__(pieces)
        self._callback = callback
        self._step_count = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self) -> tuple[tuple[int, int], int, int] | None:
        """Perform one BPE training step.

        Returns
        -------
        tuple or None
            ``(pair, rank, frequency)`` if a merge was found, or
            ``None`` if no more merges are possible.
        """
        result: tuple[tuple[int, int], int, int] | None = super().step()
        if result is not None:
            self._step_count += 1
            if self._callback is not None:
                pair, rank, freq = result
                self._callback(self._step_count, 0, pair, rank, freq)
        return result

    def train(self, n_merges: int) -> int:
        """Train for ``n_merges`` steps.

        Training stops early if no more merges are possible.

        Parameters
        ----------
        n_merges : int
            Number of merges to learn.

        Returns
        -------
        int
            The actual number of merges performed.
        """
        for i in range(n_merges):
            result = self.step()
            if result is None:
                return i
        return n_merges

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def merges(self) -> list[tuple[int, int]]:  # type: ignore[override]
        """The learned merge pairs."""
        return super().merges

    @property
    def n_merges(self) -> int:  # type: ignore[override]
        """Number of merges learned so far."""
        return super().n_merges

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the trained model to a ``.tbm`` file.

        Parameters
        ----------
        path : str
            Output path (``.tbm`` appended if missing).
        """
        save_model(path, self.merges)
