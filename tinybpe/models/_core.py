from .._abc import ABCTokenizer
from .. import bpe
import regex as re
from typing import Callable, Optional


class Encoding(ABCTokenizer):
    def __init__(self, merges: list[tuple[int, int]],
                 pat_str: str,
                 *,
                 remaps: Optional[list[int]] = None,
                 special_tokens: Optional[dict[str, int]] = None,
                 ):
        if remaps is None:
            self.bytes_remap = None
            self.inverse_bytes_remap = None
        else:
            self.bytes_remap = bpe.BytesRemap(remaps)
            _inverse_remaps = [0] * len(remaps)
            for i, v in enumerate(remaps):
                _inverse_remaps[v] = i
            self.inverse_bytes_remap = bpe.BytesRemap(_inverse_remaps)

        if special_tokens is None:
            self._enc = bpe.Tokenizer(merges)
            self.special_tokens = None
            self.special_pattern = None
        else:
            self.special_tokens = special_tokens
            if self.bytes_remap is None:
                _special_tokens = {k.encode("utf-8"): v for k, v in special_tokens.items()}
            else:
                _special_tokens = {self.bytes_remap(k.encode("utf-8")): v for k, v in special_tokens.items()}
            self._enc = bpe.Tokenizer(merges, _special_tokens)
            self.special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"

        self.compiled_pattern = re.compile(pat_str)

    def encode_ordinary(self, text: str) -> list[int]:
        text_chunks = re.findall(self.compiled_pattern, text)
        chunk_bytes = [ch.encode("utf-8") for ch in text_chunks]
        if self.bytes_remap is not None:
            chunk_bytes = list(map(self.bytes_remap, chunk_bytes))
        ids = sum(list(map(self._enc.encode, chunk_bytes)), [])
        return ids

    def encode(self, text: str) -> list[int]:
        special_chunks = re.split(self.special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        _bytes = self._enc.decode(ids)
        if self.inverse_bytes_remap is not None:
            _bytes = self.inverse_bytes_remap(_bytes)
        return _bytes.decode("utf-8")

    def stream_decode(self, callback_fn: Callable[[str], None]) -> Callable[[int], None]:
        pass

    def stream_decode_cache_clean(self):
        pass

    @property
    def merges(self) -> list[tuple[int, int]]:
        return self._enc.merges

    @property
    def vocab(self) -> dict[int, bytes]:
        if self.inverse_bytes_remap is not None:
            _vocab = {k: self.inverse_bytes_remap(v) for k, v in self._enc.vocab.items()}
        else:
            _vocab = self._enc.vocab
        return _vocab
