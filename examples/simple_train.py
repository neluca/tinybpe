from tinybpe import SimpleTrainer

text = open("taylorswift.txt", "r", encoding="utf-8").read()
train = SimpleTrainer(text)
vocab_size = 768
merges_size = vocab_size - 256
for _ in range(merges_size):
    pair, rank, freq = train.step()
    print(f"{pair} -> {rank} ({freq})")

train.save("simple")
