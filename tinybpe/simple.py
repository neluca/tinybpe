"""High-level Python API for the TinyBPE trainer.

Provides :class:`Trainer` (``SimpleTrainer``) which wraps the C-level
``bpe.Trainer`` with text encoding and an optional preprocessing callback.
"""

from __future__ import annotations

from typing import Callable

from tinybpe import bpe
from tinybpe._model_io import save_bpe_model


class Trainer(bpe.Trainer):
    """A simple Byte-Pair-Encoding trainer.

    This extends the C-level ``bpe.Trainer`` with automatic UTF-8 text
    encoding and optional preprocessing.

    Parameters
    ----------
    text : str
        The training text (will be UTF-8 encoded if no preprocess is given).
    preprocess : Callable[[str], list[Union[bytes, bytearray]]], optional
        Optional function that takes the raw text and returns a list of
        bytes or bytearray chunks for training. Use this for regex
        pre-tokenization.
    callback : Callable[[int, int, tuple[int, int], int, int], None], optional
        Optional progress callback called after each training step.
        Receives ``(step, total, pair, rank, frequency)`` where ``total``
        may be 0 if unknown.

    Examples
    --------
    Basic training:

    >>> trainer = Trainer("hello world hello")
    >>> trainer.step()
    ((104, 101), 256, 2)

    Training with regex preprocessing:

    >>> import regex as re
    >>> pat = re.compile(r"\\w+|\\s+")
    >>> def preprocess(text):
    ...     return [chunk.encode() for chunk in pat.findall(text)]
    >>> trainer = Trainer(text, preprocess=preprocess)

    Training with progress reporting:

    >>> def on_step(step, total, pair, rank, freq):
    ...     print(f"Step {step}: {pair} -> {rank}")
    >>> trainer = Trainer("hello world", callback=on_step)
    """

    def __init__(
        self,
        text: str,
        preprocess: Callable[[str], list[bytes | bytearray]] | None = None,
        *,
        callback: Callable[[int, int, tuple[int, int], int, int], None] | None = None,
    ) -> None:
        if preprocess is None:
            text_bytes_list = [text.encode("utf-8")]
        else:
            text_bytes_list = preprocess(text)

        super().__init__(text_bytes_list)
        self._callback = callback
        self._step_count = 0

    def step(self) -> tuple[tuple[int, int], int, int] | None:
        """Perform one BPE training step.

        Returns
        -------
        tuple or None
            ``(pair, rank, frequency)`` if a merge was found, or ``None``
            if no more merges are possible.
        """
        result = super().step()
        if result is not None and self._callback is not None:
            self._step_count += 1
            pair, rank, freq = result
            self._callback(self._step_count, 0, pair, rank, freq)
        elif result is not None:
            self._step_count += 1
        return result

    def train(self, n_merges: int) -> int:
        """Train for ``n_merges`` steps.

        Parameters
        ----------
        n_merges : int
            Number of merges to learn. Training stops early if no
            more merges are possible.

        Returns
        -------
        int
            The actual number of merges performed.

        Examples
        --------
        >>> trainer = Trainer("hello world hello world")
        >>> n = trainer.train(5)
        >>> print(trainer.n_merges)
        5
        """
        for i in range(n_merges):
            result = self.step()
            if result is None:
                return i
        return n_merges

    def save(self, file_prefix: str) -> None:
        """Save the trained model to ``<file_prefix>.tinymodel``.

        Parameters
        ----------
        file_prefix : str
            Path prefix for the output file.
        """
        save_bpe_model(file_prefix, self.merges)
