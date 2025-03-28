from tinybpe import Trainer, load_bpe_file

merges = load_bpe_file("simple.bpe")
text = open("taylorswift.txt", "r", encoding="utf-8").read()
train = Trainer(text)
train.load_merges(merges)

for _ in range(40):
    pair, token_id, rank = train.step()
    print(f"{pair} -> {token_id} ({rank})")

train.save("simple_c")
