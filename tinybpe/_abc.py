from abc import ABC, abstractmethod
from ._utils import _save_bpe_merges, _save_bpe_vocab
from typing import Callable


class ABCTokenizer(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("Subclasses must implement the \"encode\" method.")

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError("Subclasses must implement the \"decode\" method.")

    @abstractmethod
    def stream_decode(self, callback_fn: Callable[[str], None]) -> Callable[[int], None]:
        raise NotImplementedError("Subclasses must implement the \"stream_decode\" method.")

    @abstractmethod
    def stream_decode_cache_clean(self):
        raise NotImplementedError("Subclasses must implement the \"stream_decode_cache_clean\" method.")

    @property
    def merges(self) -> list[tuple[int, int]]:
        raise NotImplementedError("Subclasses must implement the \"merges\" getter method.")

    @property
    def vocab(self) -> dict[int, bytes]:
        raise NotImplementedError("Subclasses must implement the \"vocab\" getter method.")

    @property
    def n_vocab(self) -> int:
        return len(self.vocab)

    def save(self, file_prefix: str) -> None:
        _save_bpe_merges(file_prefix, self.merges)

    def save_vocab(self, file_prefix: str) -> None:
        _save_bpe_vocab(file_prefix, self.vocab)
