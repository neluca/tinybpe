from ._core import Encoding
from .._utils import load_bpe_file, load_bpe_remaps
from pathlib import Path

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}


class GPT4Tokenizer(Encoding):
    def __init__(self):
        merges = load_bpe_file(str(Path(__file__).parent.absolute().joinpath("cl100k_base.tinybpe")))
        remaps = load_bpe_remaps(str(Path(__file__).parent.absolute().joinpath("cl100k_base.remaps")))
        super().__init__(merges, GPT4_SPLIT_PATTERN, remaps=remaps, special_tokens=GPT4_SPECIAL_TOKENS)
