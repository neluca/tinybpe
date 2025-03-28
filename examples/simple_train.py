from tinybpe import Trainer

text = open("taylorswift.txt", "r", encoding="utf-8").read()
train = Trainer(text)
vocab_size = 768
merges_size = vocab_size - 256
for _ in range(vocab_size):
    pair, token_id, rank = train.step()
    print(f"{pair} -> {token_id} ({rank})")

train.save("simple")
