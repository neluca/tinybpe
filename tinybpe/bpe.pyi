"""
This is a Python-C-Extension module that implements the core algorithm of BPE (Byte-Pair-Encoding).
"""

from typing import Optional, Union


class Tokenizer:
    """ This is a BPE tokenizer implemented in C . """

    def __init__(self, merges: list[tuple[int, int]], special_tokens: Optional[dict[bytes, int]] = None): ...

    @property
    def merges(self) -> list[tuple[int, int]]: ...

    @property
    def vocab(self) -> dict[int, bytes]: ...

    @property
    def size(self) -> int: ...

    def encode(self, text_bytes: bytes) -> list[int]: ...

    def decode(self, ids: list[int]) -> bytes: ...

    def cache_decode(self, token_id: int) -> Union[bytes, None]: ...

    def cache_clean(self): ...


class Trainer:
    """ This is a BPE trainer implemented in C . """

    def __init__(self, list_bytes: list[bytes]): ...

    @property
    def merges_size(self) -> int: ...

    @property
    def merges(self) -> list[tuple[int, int]]: ...

    def step(self) -> Union[tuple[tuple[int, int], int, int], None]:
        """ :returns pair, rank, frequency. """

    ...

    def load_merges(self, merges: list[tuple[int, int]]) -> None: ...


class BytesRemap:
    """ This is a bytes(0 - 255) remapper. """

    def __init__(self, _remap: list[int]): ...

    def __call__(self, _bytes: bytes) -> bytes: ...
