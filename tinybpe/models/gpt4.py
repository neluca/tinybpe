from .._abc import ABCTokenizer
import regex as re
from .. import bpe

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}


class GPT4Tokenizer(ABCTokenizer):
    def __init__(self):
        self.compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)
        _special_tokens = {k.encode("utf-8"): v for k, v in GPT4_SPECIAL_TOKENS.items()}
