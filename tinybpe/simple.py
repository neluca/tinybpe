from tinybpe import bpe, save_bpe_model
from typing import Callable, Optional, Union


class Trainer(bpe.Trainer):
    """ A simple Byte-Pair-Encoding trainer """

    def __init__(self, text: str, preprocess: Optional[Callable[[str], list[Union[bytes, bytearray]]]] = None):
        if preprocess is None:
            text_bytes_list = [text.encode("utf-8")]
        else:
            text_bytes_list = preprocess(text)

        super().__init__(text_bytes_list)

    def save(self, file_prefix: str) -> None:
        save_bpe_model(file_prefix, self.merges)
