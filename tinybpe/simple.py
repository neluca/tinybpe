from ._utils import _save_bpe_merges
from . import bpe
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
