from tinybpe import Tokenizer, load_bpe_file

merges = load_bpe_file("simple.tinybpe")

tokenizer = Tokenizer(merges)
s = "hello world"
ids = tokenizer.encode(s)
print(ids)
s2 = tokenizer.decode(ids)
print(s2)
print(tokenizer.n_vocab)
tokenizer.save_vocab("simple")
