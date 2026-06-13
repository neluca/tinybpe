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


def _find_package_file(rel_path: str) -> str:
    """Resolve a package-relative path to an absolute filesystem path.

    Handles both wheel installs (via ``importlib.resources``) and
    editable installs (``pip install -e``).

    Parameters
    ----------
    rel_path : str
        Path relative to the package root (e.g. ``"models/cl100k_base.tbm"``).

    Returns
    -------
    str
        Absolute filesystem path to the file.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found in any expected location.
    """
    import os
    from pathlib import Path as _Path

    # Try importlib.resources first (Python 3.9+)
    try:
        from importlib.resources import files as _files

        pkg_root = _files("tinybpe")
        # Models live inside the package: tinybpe/models/
        candidate: str = str(_Path(str(pkg_root)) / rel_path)
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass

    # Fallback: look relative to the package directory (works for editable installs)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(os.path.join(package_dir, rel_path))
    if os.path.isfile(candidate):
        return candidate

    # Last resort: check if rel_path itself exists (absolute or cwd-relative)
    if os.path.isfile(rel_path):
        return os.path.abspath(rel_path)

    raise FileNotFoundError(f"Model file not found: {rel_path}")


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
            # Sort by length descending so that longer tokens match before
            # shorter prefixes (e.g. "<ab>" before "<a>").
            self._special_pattern = (
                "(" + "|".join(re.escape(k) for k in sorted(special_tokens, key=len, reverse=True)) + ")"
            )

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
        """Encode text without pre-splitting on special tokens.

        Unlike :meth:`encode`, this method does not use the special token
        regex pattern to split text before encoding.  Note that special
        tokens may still be produced if the BPE merges produce them.

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
        n_special = len(self._special_tokens) if self._special_tokens is not None else 0
        return f"Tokenizer(n_vocab={self.n_vocab}, byte_remap={has_remap}, special_tokens={n_special})"

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

    @classmethod
    def from_pretrained(cls, name: str) -> Tokenizer:
        """Load a built-in model by name.

        Models ship with the package — no network download required.
        Call :func:`tinybpe.list_models` to see available names.

        Parameters
        ----------
        name : str
            Model name (e.g. ``"cl100k_base"``, ``"qwen35"``, ``"minicpm5"``).

        Returns
        -------
        Tokenizer
            Fully configured tokenizer with the model's regex pattern
            and special tokens (when applicable).

        Raises
        ------
        ValueError
            If *name* is not a known built-in model.

        Examples
        --------
        >>> from tinybpe import Tokenizer
        >>> tok = Tokenizer.from_pretrained("cl100k_base")
        >>> ids = tok.encode("hello world")
        >>> tok.decode(ids)
        'hello world'
        """
        from tinybpe._registry import _MODEL_REGISTRY, ModelInfo

        if name not in _MODEL_REGISTRY:
            from tinybpe._registry import list_models as _list_models

            available = _list_models()
            raise ValueError(f"Unknown model {name!r}. Available models: {available}")

        info: ModelInfo = _MODEL_REGISTRY[name]
        model_path = _find_package_file(info["path"])

        merges, bytes_maps = load_model(model_path)
        return cls(
            merges,
            bytes_maps=bytes_maps,
            pat_str=info.get("pat_str"),
            special_tokens=info.get("special_tokens"),
        )
