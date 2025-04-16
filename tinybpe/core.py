from tinybpe import bpe
from tinybpe._utils import save_bpe_vocab, save_bpe_model, BPEParam
from typing import Callable, Optional
import regex as re  # type: ignore


class CommonTokenizer:
    def __init__(self, merges: list[tuple[int, int]],
                 pat_str: Optional[str] = None, *,
                 special_tokens: Optional[dict[str, int]] = None
                 ):
        """ Creates a common Tokenizer object """
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
        """ Encodes a string into tokens, ignoring special tokens."""
        text_chunks = re.findall(self._compiled_pattern, text)
        chunk_bytes = [ch.encode("utf-8") for ch in text_chunks]
        ids = sum(list(map(self._enc.encode, chunk_bytes)), [])
        return ids

    def encode(self, text: str) -> list[int]:
        """ Encodes a string into tokens."""
        if self._special_pattern is None:
            return self.encode_ordinary(text)

        special_chunks = re.split(self._special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in self._special_tokens:  # type: ignore
                ids.append(self._special_tokens[part])  # type: ignore
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        """ Decodes a list of tokens into a string."""
        text_bytes = self._enc.decode(ids)
        return text_bytes.decode("utf-8")

    def stream_decode(self, callback: Callable[[str], None]) -> Callable[[int], None]:
        """ Streaming Decoding ."""
        self._enc.cache_clean()

        def _decode(token_id: int):
            text_bytes = self._enc.cache_decode(token_id)
            if text_bytes is not None:
                callback(text_bytes.decode("utf-8"))

        return _decode

    def stream_decode_cache_clean(self) -> None:
        """ Clear the cache of streaming decoding. """
        self._enc.cache_clean()

    @property
    def merges(self) -> list[tuple[int, int]]:
        """ The core parameters of the BPE encoder. """
        return self._enc.merges

    @property
    def vocab(self) -> dict[int, bytes]:
        """ Vocabulary, the core parameters of the BPE decoder. """
        return self._enc.vocab

    @property
    def n_vocab(self) -> int:
        """ The actual size of the vocabulary. """
        return self._enc.size

    def save(self, file_prefix: str) -> None:
        """ Save the parameters to the model file. """
        save_bpe_model(file_prefix, self.merges)

    def save_vocab(self, file_prefix: str) -> None:
        """ Save the vocabulary to a file. """
        save_bpe_vocab(file_prefix, self.vocab)


class Tokenizer(CommonTokenizer):
    def __init__(self, bpe_param: BPEParam,
                 pat_str: Optional[str] = None, *,
                 special_tokens: Optional[dict[str, int]] = None
                 ):
        """ Creates a Tokenizer object with bytes remapping """
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
            pat_str = r"^.*$"  # do nothing
        self._compiled_pattern = re.compile(pat_str)
        self._cache = b""

    def encode_ordinary(self, text: str) -> list[int]:
        if self._bytes_maps is None:
            return super().encode_ordinary(text)

        text_chunks = re.findall(self._compiled_pattern, text)
        chunk_bytes = [ch.encode("utf-8") for ch in text_chunks]
        chunk_bytes = list(map(self._map, chunk_bytes))  # type: ignore
        ids = sum(list(map(self._enc.encode, chunk_bytes)), [])
        return ids

    def decode(self, ids: list[int]) -> str:
        if self._bytes_maps is None:
            return super().decode(ids)

        text_bytes = self._enc.decode(ids)
        text_bytes = self._inv_map(text_bytes)  # type: ignore
        return text_bytes.decode("utf-8")

    def stream_decode(self, callback: Callable[[str], None]) -> Callable[[int], None]:
        if self._bytes_maps is None:
            return super().stream_decode(callback)

        self._cache = b""

        def _decode(token_id: int):
            text_bytes = self._enc.decode([token_id])
            text_bytes = self._inv_map(text_bytes)  # type: ignore
            text_bytes = self._cache + text_bytes
            try:
                text = text_bytes.decode("utf-8")
                self._cache = b""
                callback(text)
            except UnicodeDecodeError:
                self._cache = text_bytes

        return _decode

    def stream_decode_cache_clean(self) -> None:
        if self._bytes_maps is None:
            return super().stream_decode_cache_clean()
        self._cache = b""

    @property
    def vocab(self) -> dict[int, bytes]:
        if self._bytes_maps is None:
            return super().vocab
        return {k: self._inv_map(v) for k, v in self._enc.vocab.items()}  # type: ignore

    def save(self, file_prefix: str) -> None:
        if self._bytes_maps is None:
            return super().save(file_prefix)
        save_bpe_model(file_prefix, self.merges, self._bytes_maps)
