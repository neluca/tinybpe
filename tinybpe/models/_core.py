from .._abc import ABCTokenizer
from typing import Callable, Optional
from .. import bpe


class Encoding(ABCTokenizer):
    def __init__(self, merges: list[tuple[int, int]],
                 *,
                 pat_str: str = None,
                 remaps: Optional[list[int]] = None,
                 special_tokens: Optional[dict[str, int]] = None,
                 ):
        if special_tokens is None:
            self._enc = bpe.Tokenizer(merges)
        else:
            _special_tokens = {k.encode("utf-8"): v for k, v in special_tokens.items()}
            self._enc = bpe.Tokenizer(merges, _special_tokens)

        # if remaps is None:
        #     self._tokenize = lambda s: [s.encode("utf-8")]
        # else:
        #     self._tokenize = remaps

    def encode(self, text: str) -> list[int]:
        # text_bytes_list = self._tokenize(text)
        text_bytes_list = [text.encode("utf-8")]
        ids = sum(list(map(self._enc.encode, text_bytes_list)), [])
        return ids

    def decode(self, ids: list[int]) -> str:
        text_bytes = self._enc.decode(ids)
        return text_bytes.decode("utf-8")

    def stream_decode(self, callback_fn: Callable[[str], None]) -> Callable[[int], None]:
        pass

    def stream_decode_cache_clean(self):
        pass

    @property
    def merges(self) -> list[tuple[int, int]]:
        return self._enc.merges

    @property
    def vocab(self) -> dict[int, bytes]:
        return self._enc.vocab

    @property
    def n_vocab(self) -> int:
        return self._enc.size
