from tinybpe import Tokenizer, load_bpe_file

merges = load_bpe_file("simple.tinybpe")

tokenizer = Tokenizer(merges)

s = "hello world 你好世界"


def cb_print(s_):
    print(s_, end="")


decode = tokenizer.stream_decode(cb_print)
ids = tokenizer.encode(s)
print(ids)
for i in ids:
    decode(i)

print("")
