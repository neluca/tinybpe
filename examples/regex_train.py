import regex as re
from tinybpe import Trainer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Preprocess:
    def __init__(self, pattern=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

    def __call__(self, text_):
        text_chunks = re.findall(self.compiled_pattern, text_)
        ids = [ch.encode("utf-8") for ch in text_chunks]
        return ids


process = Preprocess()

text = open("taylorswift.txt", "r", encoding="utf-8").read()
train = Trainer(text, process)
vocab_size = 768
merges_size = vocab_size - 256
for _ in range(merges_size):
    pair, rank, freq = train.step()
    print(f"{pair} -> {rank} ({freq})")

train.save("regex")
