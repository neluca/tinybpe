from . import bpe
from ._utils import _save_bpe_merges, _save_bpe_vocab, load_bpe_file
from typing import Callable, Optional


class Trainer(bpe.Trainer):
    def __init__(self, text: str, preprocess: Optional[Callable[[str], list[bytes | bytearray]]] = None):
        if preprocess is None:
            text_bytes_list = [text.encode("utf-8")]
        else:
            text_bytes_list = preprocess(text)

        super().__init__(text_bytes_list)

    def save(self, file_prefix: str) -> None:
        _save_bpe_merges(file_prefix, self.merges)


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]], special_tokens: Optional[dict[str, int]] = None,
                 pre_tokenize: Optional[Callable[[str], list[bytes]]] = None):
        if special_tokens is None:
            self._enc = bpe.Tokenizer(merges)
        else:
            _special_tokens = {k.encode("utf-8"): v for k, v in special_tokens.items()}
            self._enc = bpe.Tokenizer(merges, _special_tokens)

        if pre_tokenize is None:
            self._tokenize = lambda s: [s.encode("utf-8")]
        else:
            self._tokenize = pre_tokenize

    def encode(self, text: str) -> list[int]:
        text_bytes_list = self._tokenize(text)
        ids = sum(list(map(self._enc.encode, text_bytes_list)), [])
        return ids

    def decode(self, ids: list[int]) -> str:
        text_bytes = self._enc.decode(ids)
        return text_bytes.decode("utf-8")

    def stream_decode(self, callback_fn: Callable[[str], None]) -> Callable[[int], None]:
        self._enc.cache_clean()

        def _decode(token_id: int):
            text_bytes = self._enc.cache_decode(token_id)
            if text_bytes is not None:
                callback_fn(text_bytes.decode("utf-8"))

        return _decode

    def stream_decode_cache_clean(self):
        self._enc.cache_clean()

    @property
    def merges(self) -> list[tuple[int, int]]:
        return self._enc.merges

    @property
    def vocab(self) -> dict[int, bytes]:
        return self._enc.vocab

    @property
    def n_vocab(self) -> int:
        return self._enc.size

    def save(self, file_prefix: str) -> None:
        _save_bpe_merges(file_prefix, self.merges)

    def save_vocab(self, file_prefix: str) -> None:
        _save_bpe_vocab(file_prefix, self.vocab)
