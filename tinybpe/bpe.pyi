"""
This is a Python-C-Extension module that implements the core algorithm of BPE (Byte-Pair-Encoding).
"""

from typing import Optional


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]], special_tokens: Optional[dict[bytes, int]] = None):
        ...

    @property
    def merges(self) -> list[tuple[int, int]]:
        ...

    @property
    def vocab(self) -> dict[int, bytes]:
        ...

    @property
    def size(self) -> int:
        ...

    def encode(self, text_bytes: bytes) -> list[int]:
        ...

    def decode(self, ids: list[int]) -> bytes:
        ...

    def cache_decode(self, token_id: int) -> bytes | None:
        ...

    def cache_clean(self):
        ...


class Trainer:
    def __init__(self, text_bytes_list: list[bytes]):
        ...

    @property
    def merges_size(self) -> int:
        ...

    @property
    def merges(self) -> list[tuple[int, int]]:
        ...

    def step(self) -> tuple[tuple[int, int], int, int] | None:
        ...

    def load_merges(self, merges: list[tuple[int, int]]) -> None:
        ...
