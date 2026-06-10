"""High-level Python API for the TinyBPE tokenizer.

Provides :class:`CommonTokenizer` and :class:`Tokenizer` classes that wrap
the C extension with regex pre-tokenization, special token handling,
byte remapping, and streaming decode support.
"""

from __future__ import annotations

from typing import Callable

import regex as re

import tinybpe.bpe as bpe
from tinybpe._model_io import BPEParam, save_bpe_model, save_bpe_vocab


class CommonTokenizer:
    """A byte-level BPE tokenizer with regex pre-tokenization support.

    This is the simpler tokenizer variant without byte remapping.
    For tokenizers that need byte remapping (e.g., tiktoken compatibility),
    use :class:`Tokenizer` instead.

    Parameters
    ----------
    merges : list[tuple[int, int]]
        BPE merge pairs defining the vocabulary.
    pat_str : str, optional
        Regex pattern for pre-tokenization. Defaults to ``r"^.*$"``
        (no splitting).
    special_tokens : dict[str, int], optional
        Mapping from special token strings to their IDs.
    """

    def __init__(
        self,
        merges: list[tuple[int, int]],
        pat_str: str | None = None,
        *,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        """Initialize the tokenizer.

        Raises
        ------
        ValueError
            If the merges list is invalid.
        TypeError
            If merges is not a list of integer pairs.
        """
        if special_tokens is None:
            self._enc = bpe.Tokenizer(merges)
            self._special_tokens = None
            self._special_pattern = None
        else:
            _special_tokens = {k.encode("utf-8"): v for k, v in special_tokens.items()}
            self._enc = bpe.Tokenizer(merges, _special_tokens)
            self._special_tokens = special_tokens
            self._special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"

        if pat_str is None:
            pat_str = r"^.*$"  # do nothing
        self._compiled_pattern = re.compile(pat_str)

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text into token IDs, ignoring special tokens.

        Parameters
        ----------
        text : str
            The input text to encode.

        Returns
        -------
        list[int]
            List of token IDs.
        """
        text_chunks = re.findall(self._compiled_pattern, text)
        chunk_bytes = [ch.encode("utf-8") for ch in text_chunks]
        ids = [token_id for chunk in chunk_bytes for token_id in self._enc.encode(chunk)]
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs, including special tokens.

        Parameters
        ----------
        text : str
            The input text to encode.

        Returns
        -------
        list[int]
            List of token IDs.
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
        return text_bytes.decode("utf-8")

    def stream_decode(self, callback: Callable[[str], None]) -> Callable[[int], None]:
        """Create a streaming decoder that calls ``callback`` for each decoded fragment.

        Stream decoding processes one token ID at a time and handles
        partial UTF-8 sequences by caching incomplete bytes.

        Parameters
        ----------
        callback : Callable[[str], None]
            Function called with each decoded text fragment.

        Returns
        -------
        Callable[[int], None]
            A function that accepts one token ID at a time.
        """
        self._enc.cache_clean()

        def _decode(token_id: int) -> None:
            text_bytes = self._enc.cache_decode(token_id)
            if text_bytes is not None:
                callback(text_bytes.decode("utf-8"))

        return _decode

    def stream_decode_cache_clean(self) -> None:
        """Clear the internal cache used by stream decoding."""
        self._enc.cache_clean()

    @property
    def merges(self) -> list[tuple[int, int]]:
        """The BPE merge pairs that define the vocabulary."""
        return self._enc.merges

    @property
    def vocab(self) -> dict[int, bytes]:
        """Vocabulary mapping token IDs to byte sequences."""
        return self._enc.vocab

    @property
    def n_vocab(self) -> int:
        """The total vocabulary size (256 + n_merges + n_special_tokens)."""
        return self._enc.n_vocab

    def save(self, file_prefix: str) -> None:
        """Save model parameters to ``<file_prefix>.tinymodel``.

        Only the BPE merge pairs (and byte remapping, if present) are saved.
        The regex pattern and special tokens configuration are NOT preserved.
        Use :func:`tinybpe.save_bpe_model` directly for full control.

        Parameters
        ----------
        file_prefix : str
            Path prefix for the output file.
        """
        save_bpe_model(file_prefix, self.merges)

    def save_vocab(self, file_prefix: str) -> None:
        """Save vocabulary to ``<file_prefix>.vocab``.

        Parameters
        ----------
        file_prefix : str
            Path prefix for the output file.
        """
        save_bpe_vocab(file_prefix, self.vocab)


class Tokenizer(CommonTokenizer):
    """A BPE tokenizer with byte remapping support.

    This extends :class:`CommonTokenizer` with byte remapping,
    which is necessary for compatibility with tiktoken models
    where byte values are permuted before encoding.

    Parameters
    ----------
    bpe_param : BPEParam
        Model parameters including merges and optional byte remapping.
    pat_str : str, optional
        Regex pattern for pre-tokenization.
    special_tokens : dict[str, int], optional
        Mapping from special token strings to their IDs.
    """

    def __init__(
        self,
        bpe_param: BPEParam,
        pat_str: str | None = None,
        *,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        """Initialize the tokenizer with optional byte remapping.

        Raises
        ------
        ValueError
            If the merges or byte maps are invalid.
        """
        if bpe_param.bytes_maps is None:
            self._bytes_maps = None
            self._map = None
            self._inv_map = None
            super().__init__(bpe_param.merges, pat_str, special_tokens=special_tokens)
            return

        self._bytes_maps = bpe_param.bytes_maps
        self._map = bpe.BytesRemap(self._bytes_maps)
        inv_maps = [0] * len(self._bytes_maps)
        for i, v in enumerate(self._bytes_maps):
            inv_maps[v] = i
        self._inv_map = bpe.BytesRemap(inv_maps)

        if special_tokens is None:
            self._enc = bpe.Tokenizer(bpe_param.merges)
            self._special_tokens = None
            self._special_pattern = None
        else:
            _special_tokens = {self._map(k.encode("utf-8")): v for k, v in special_tokens.items()}
            self._enc = bpe.Tokenizer(bpe_param.merges, _special_tokens)
            self._special_tokens = special_tokens
            self._special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"

        if pat_str is None:
            pat_str = r"^.*$"
        self._compiled_pattern = re.compile(pat_str)
        self._cache: bytes = b""

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text with byte remapping applied before encoding.

        Parameters
        ----------
        text : str
            The input text to encode.

        Returns
        -------
        list[int]
            List of token IDs.
        """
        if self._bytes_maps is None:
            return super().encode_ordinary(text)

        text_chunks = re.findall(self._compiled_pattern, text)
        chunk_bytes = [ch.encode("utf-8") for ch in text_chunks]
        chunk_bytes = list(map(self._map, chunk_bytes))  # type: ignore[arg-type]
        ids = [token_id for chunk in chunk_bytes for token_id in self._enc.encode(chunk)]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs with inverse byte remapping.

        Parameters
        ----------
        ids : list[int]
            The token IDs to decode.

        Returns
        -------
        str
            The decoded text.
        """
        if self._bytes_maps is None:
            return super().decode(ids)

        text_bytes = self._enc.decode(ids)
        text_bytes = self._inv_map(text_bytes)  # type: ignore[misc]
        return text_bytes.decode("utf-8")

    def stream_decode(self, callback: Callable[[str], None]) -> Callable[[int], None]:
        """Create a streaming decoder with byte remapping.

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
            return super().stream_decode(callback)

        self._cache = b""

        def _decode(token_id: int) -> None:
            text_bytes = self._enc.decode([token_id])
            text_bytes = self._inv_map(text_bytes)  # type: ignore[misc]
            text_bytes = self._cache + text_bytes
            try:
                text = text_bytes.decode("utf-8")
                self._cache = b""
                callback(text)
            except UnicodeDecodeError:
                self._cache = text_bytes

        return _decode

    def stream_decode_cache_clean(self) -> None:
        """Clear the stream decoding cache."""
        if self._bytes_maps is None:
            return super().stream_decode_cache_clean()
        self._cache = b""

    @property
    def vocab(self) -> dict[int, bytes]:
        """Vocabulary with inverse byte remapping applied."""
        if self._bytes_maps is None:
            return super().vocab
        return {k: self._inv_map(v) for k, v in self._enc.vocab.items()}  # type: ignore[misc]

    def save(self, file_prefix: str) -> None:
        """Save model with byte remapping to ``<file_prefix>.tinymodel``.

        Parameters
        ----------
        file_prefix : str
            Path prefix for the output file.
        """
        if self._bytes_maps is None:
            return super().save(file_prefix)
        save_bpe_model(file_prefix, self.merges, self._bytes_maps)
