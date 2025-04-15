from tinybpe import Tokenizer, load_bpe_model

model = load_bpe_model("simple-chinese.tinymodel")
tokenizer = Tokenizer(model)
s1 = "他是一个独自一人划着小船在墨西哥湾大海流打鱼的老人"
ids = tokenizer.encode(s1)
print(ids)
s2 = tokenizer.decode(ids)
print(s2)
tokenizer.save_vocab("simple-chinese")
