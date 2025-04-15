import regex as re
from tinybpe import SimpleTrainer

SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


class Preprocess:
    def __init__(self):
        self.pattern = SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)

    def __call__(self, text_):
        text_chunks = re.findall(self.compiled_pattern, text_)
        ids = [ch.encode("utf-8") for ch in text_chunks]
        return ids


proc = Preprocess()
text = open("the-old-man-and-the-sea.txt", "r", encoding="utf-8").read()
trainer = SimpleTrainer(text, proc)
vocab_size = 1000
merges_size = vocab_size - 256
for _ in range(merges_size):
    pair, rank, freq = trainer.step()
    print(f"{pair} -> {rank} ({freq})")

trainer.save("regex")
