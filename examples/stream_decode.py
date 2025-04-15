from tinybpe import Tokenizer, load_bpe_model

model = load_bpe_model("simple.tinymodel")
tokenizer = Tokenizer(model)
tokenizer_2 = Tokenizer(model)

s1 = "hello world 你好世界"


def cb_print(text):
    print(text, end="")


g_text = ""


def cb(text):
    global g_text
    g_text += text


decode_1 = tokenizer.stream_decode(cb_print)
decode_2 = tokenizer_2.stream_decode(cb)
ids = tokenizer.encode(s1)
print(ids)
for i in ids:
    decode_1(i)
    decode_2(i)
print("")
print(g_text)
