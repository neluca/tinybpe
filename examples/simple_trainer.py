from tinybpe import SimpleTrainer

text = open("the-old-man-and-the-sea.txt", "r", encoding="utf-8").read()
trainer = SimpleTrainer(text)
vocab_size = 1000
merges_size = vocab_size - 256
for _ in range(merges_size):
    pair, rank, freq = trainer.step()
    print(f"{pair} -> {rank} ({freq})")

print(trainer.merges)
print(trainer.merges_size)
trainer.save("simple")
