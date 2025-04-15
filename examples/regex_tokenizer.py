from tinybpe import Tokenizer, load_bpe_model

SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

vocab_size = 1000
special_tokens = {
    "<eot>": vocab_size,
    "<fim_prefix>": vocab_size + 1,
    "<fim_middle>": vocab_size + 2,
    "<fim_suffix>": vocab_size + 3,
    "<eop>": vocab_size + 4,
}
model = load_bpe_model("regex.tinymodel")
tokenizer = Tokenizer(model, SPLIT_PATTERN, special_tokens=special_tokens)
s = "<fim_prefix> hello world 中文<eot><fim_suffix>"
ids = tokenizer.encode(s)
print(ids)
s2 = tokenizer.decode(ids)
print(s2)
print(tokenizer.n_vocab)
tokenizer.save_vocab("regex")
