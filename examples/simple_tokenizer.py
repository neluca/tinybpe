from tinybpe import Tokenizer, load_bpe_model

model = load_bpe_model("simple.tinymodel")
tokenizer = Tokenizer(model)
s1 = "hello world, old man !"
ids = tokenizer.encode(s1)
print(ids)
s2 = tokenizer.decode(ids)
print(s2)
tokenizer.save_vocab("simple")
