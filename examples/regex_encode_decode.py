import regex as re
from tinybpe import Tokenizer, load_bpe_file


class PreTokenize:
    def __init__(self, special_tokens_):
        self.special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens_) + ")"

    def __call__(self, text_):
        text_chunks = re.split(self.special_pattern, text_)
        ids_ = [ch.encode("utf-8") for ch in text_chunks]
        return ids_


vocab_size = 768

special_tokens = {
    "<eot>": vocab_size,
    "<fim_prefix>": vocab_size + 1,
    "<fim_middle>": vocab_size + 2,
    "<fim_suffix>": vocab_size + 3,
    "<eop>": vocab_size + 4,
}
pre_tokenize = PreTokenize(special_tokens)

merges = load_bpe_file("regex.bpe")

tokenizer = Tokenizer(merges, special_tokens=special_tokens, pre_tokenize=pre_tokenize)

s = "<fim_prefix> hello world 中文<eot><fim_suffix>"
ids = tokenizer.encode(s)
print(ids)
s2 = tokenizer.decode(ids)
print(s2)
print(tokenizer.n_vocab)
tokenizer.save_vocab("regex")
