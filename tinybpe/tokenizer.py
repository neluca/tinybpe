"""High-level Python tokenizer with regex pre-tokenization and streaming decode.

Provides the :class:`Tokenizer` class which wraps the C-level
``bpe.Tokenizer`` with:

- Regex pre-tokenization (splitting text into chunks before encoding)
- Optional byte remapping (for tiktoken compatibility)
- Special token handling
- Streaming decode with UTF-8 boundary caching
"""

from __future__ import annotations

from typing import Callable

import regex as re

import tinybpe.bpe as bpe
from tinybpe._model_io import load_model, save_model, save_vocab


class Tokenizer:
    """A byte-level BPE tokenizer.

    Wraps the C extension with regex pre-tokenization, special token
    handling, byte remapping, and streaming decode support.

    Parameters
    ----------
    merges : list[tuple[int, int]]
        BPE merge pairs defining the vocabulary.
    bytes_maps : list[int] or None
        Optional byte remapping table of 256 integers.  When provided,
        input bytes are permuted before encoding and inverse-permuted
        after decoding (required for tiktoken-compatible models).
    pat_str : str or None
        Regex pattern for pre-tokenization.  Defaults to ``r"^.*$"``
        (no splitting — treat entire input as one chunk).
    special_tokens : dict[str, int] or None
        Mapping from special token strings to their IDs.

    Examples
    --------
    >>> tok = Tokenizer(merges, pat_str=r"\\w+|\\s+")
    >>> ids = tok.encode("hello world")
    >>> text = tok.decode(ids)
    """

    def __init__(
        self,
        merges: list[tuple[int, int]],
        *,
        bytes_maps: list[int] | None = None,
        pat_str: str | None = None,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        # ---- byte remapping ----
        if bytes_maps is not None:
            self._bytes_maps: list[int] | None = bytes_maps
            self._map: bpe.BytesRemap | None = bpe.BytesRemap(bytes_maps)
            inv_maps = [0] * 256
            for i, v in enumerate(bytes_maps):
                inv_maps[v] = i
            self._inv_map: bpe.BytesRemap | None = bpe.BytesRemap(inv_maps)
        else:
            self._bytes_maps = None
            self._map = None
            self._inv_map = None

        # ---- special tokens ----
        if special_tokens is None:
            self._special_tokens: dict[str, int] | None = None
            self._special_pattern: str | None = None
            _mapped: dict[bytes, int] | None = None
        else:
            self._special_tokens = special_tokens
            self._special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"

            if self._bytes_maps is None:
                _mapped = {k.encode("utf-8"): v for k, v in special_tokens.items()}
            else:
                assert self._map is not None
                _mapped = {self._map(k.encode("utf-8")): v for k, v in special_tokens.items()}
        self._enc = bpe.Tokenizer(merges, _mapped)

        # ---- pre-tokenization pattern ----
        if pat_str is None:
            # Match the entire input including newlines (DOTALL).
            pat_str = r"(?s)^.*$"
        self._compiled_pattern = re.compile(pat_str)

        # ---- streaming decode state ----
        self._stream_cache: bytes = b""

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, ignoring special tokens.

        Parameters
        ----------
        text : str
            The input text to encode.

        Returns
        -------
        list[int]
            Token ID sequence.
        """
        chunks = re.findall(self._compiled_pattern, text)
        chunk_bytes = [ch.encode("utf-8") for ch in chunks]

        if self._bytes_maps is not None:
            assert self._map is not None
            chunk_bytes = [self._map(b) for b in chunk_bytes]
        ids: list[int] = []
        for chunk in chunk_bytes:
            ids.extend(self._enc.encode(chunk))
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text, respecting special tokens.

        Parameters
        ----------
        text : str
            The input text to encode.

        Returns
        -------
        list[int]
            Token ID sequence (including special token IDs).
        """
        if self._special_pattern is None:
            return self.encode_ordinary(text)

        special_chunks = re.split(self._special_pattern, text)
        ids: list[int] = []
        for part in special_chunks:
            if part in self._special_tokens:  # type: ignore[operator]
                ids.append(self._special_tokens[part])  # type: ignore[index]
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string.

        Parameters
        ----------
        ids : list[int]
            The token IDs to decode.

        Returns
        -------
        str
            The decoded text.
        """
        text_bytes = self._enc.decode(ids)

        if self._bytes_maps is not None:
            assert self._inv_map is not None
            text_bytes = self._inv_map(text_bytes)
        return text_bytes.decode("utf-8")

    # ------------------------------------------------------------------
    # Streaming decode
    # ------------------------------------------------------------------

    def stream_decode(self, callback: Callable[[str], None]) -> Callable[[int], None]:
        """Create a streaming decoder.

        Processes one token ID at a time and calls ``callback`` with
        each complete text fragment.  Handles partial UTF-8 sequences
        by caching incomplete bytes across calls.

        Parameters
        ----------
        callback : Callable[[str], None]
            Function called with each decoded text fragment.

        Returns
        -------
        Callable[[int], None]
            A function that accepts one token ID at a time.
        """
        if self._bytes_maps is None:
            self._enc.cache_clean()

            def _decode(token_id: int) -> None:
                text_bytes = self._enc.cache_decode(token_id)
                if text_bytes is not None:
                    callback(text_bytes.decode("utf-8"))

            return _decode

        # With byte remapping: use Python-level cache
        self._enc.cache_clean()
        self._stream_cache = b""

        def _decode_remap(token_id: int) -> None:
            assert self._inv_map is not None
            text_bytes = self._enc.decode([token_id])
            text_bytes = self._inv_map(text_bytes)
            text_bytes = self._stream_cache + text_bytes
            try:
                text = text_bytes.decode("utf-8")
                self._stream_cache = b""
                callback(text)
            except UnicodeDecodeError:
                self._stream_cache = text_bytes

        return _decode_remap

    def stream_decode_reset(self) -> None:
        """Clear the streaming decode cache."""
        self._enc.cache_clean()
        self._stream_cache = b""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation."""
        has_remap = self._bytes_maps is not None
        has_special = self._special_tokens is not None
        n_special = len(self._special_tokens) if has_special else 0
        return (
            f"Tokenizer(n_vocab={self.n_vocab}, byte_remap={has_remap}, "
            f"special_tokens={n_special})"
        )

    @property
    def merges(self) -> list[tuple[int, int]]:
        """The BPE merge pairs that define the vocabulary."""
        return self._enc.merges

    @property
    def vocab(self) -> dict[int, bytes]:
        """Vocabulary mapping token IDs to byte sequences."""
        if self._bytes_maps is None:
            return self._enc.vocab
        assert self._inv_map is not None
        return {k: self._inv_map(v) for k, v in self._enc.vocab.items()}

    @property
    def n_vocab(self) -> int:
        """Total vocabulary size (256 + n_merges + n_special_tokens)."""
        return self._enc.n_vocab

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the model to a ``.tbm`` file.

        Only the BPE merge pairs and byte remapping (if present) are
        saved.  The regex pattern and special tokens are NOT preserved.

        Parameters
        ----------
        path : str
            Output path (``.tbm`` appended if missing).
        """
        save_model(path, self.merges, self._bytes_maps)

    def save_vocab(self, path: str) -> None:
        """Save the vocabulary to a ``.vocab`` file.

        Parameters
        ----------
        path : str
            Output path (``.vocab`` appended if missing).
        """
        save_vocab(path, self.vocab)

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        pat_str: str | None = None,
        special_tokens: dict[str, int] | None = None,
    ) -> Tokenizer:
        """Create a Tokenizer from a ``.tbm`` model file.

        Parameters
        ----------
        path : str
            Path to the ``.tbm`` file.
        pat_str : str or None
            Regex pattern for pre-tokenization.
        special_tokens : dict[str, int] or None
            Mapping from special token strings to their IDs.

        Returns
        -------
        Tokenizer
            The loaded tokenizer.
        """
        merges, bytes_maps = load_model(path)
        return cls(
            merges,
            bytes_maps=bytes_maps,
            pat_str=pat_str,
            special_tokens=special_tokens,
        )
