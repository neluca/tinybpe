from tinybpe import SimpleTrainer, load_bpe_model

text = open("the-old-man-and-the-sea.txt", "r", encoding="utf-8").read()
trainer = SimpleTrainer(text)
model = load_bpe_model("simple.tinymodel")
trainer.load_merges(model.merges)
for _ in range(50):
    pair, rank, freq = trainer.step()
    print(f"{pair} -> {rank} ({freq})")

trainer.save("simple-continue")
