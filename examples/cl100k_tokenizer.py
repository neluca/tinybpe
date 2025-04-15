from tinybpe import Tokenizer, load_bpe_model

SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

model = load_bpe_model("cl100k_base.tinymodel")
tokenizer = Tokenizer(model, SPLIT_PATTERN, special_tokens=SPECIAL_TOKENS)
tokenizer.save_vocab("cl100k_base")
s1 = "hello world 你好世界 <|endoftext|>"
ids = tokenizer.encode(s1)
print(ids)
s2 = tokenizer.decode(ids)
print(s2)


def cb_print(text):
    print(text, end="")


decode = tokenizer.stream_decode(cb_print)
for i in ids:
    decode(i)
