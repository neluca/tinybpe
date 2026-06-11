"""Type stubs for the TinyBPE C extension module."""

class Trainer:
    """C-level BPE trainer.  Construct with a list of bytes/bytearray chunks."""

    merges: list[tuple[int, int]]
    n_merges: int

    def __init__(self, list_bytes: list[bytes | bytearray]) -> None: ...
    def step(self) -> tuple[tuple[int, int], int, int] | None: ...
    def load_merges(self, merges: list[tuple[int, int]]) -> None: ...

class Tokenizer:
    """C-level BPE tokenizer.  Construct with merges and optional special tokens."""

    merges: list[tuple[int, int]]
    vocab: dict[int, bytes]
    n_vocab: int

    def __init__(
        self,
        merges: list[tuple[int, int]],
        special_tokens: dict[bytes, int] | None = None,
    ) -> None: ...
    def encode(self, data: bytes) -> list[int]: ...
    def decode(self, ids: list[int]) -> bytes: ...
    def cache_decode(self, id: int) -> bytes | None: ...
    def cache_clean(self) -> None: ...

class BytesRemap:
    """Callable byte-level permutation (0-255)."""

    def __init__(self, _remap: list[int]) -> None: ...
    def __call__(self, _bytes: bytes) -> bytes: ...
