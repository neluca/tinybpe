from tinybpe import models

enc = models.GPT4Tokenizer()
s = "hello world <|endoftext|> 廖寅安"
ids = enc.encode(s)
s1 = enc.decode(ids)

print(ids)
print(s1)


def print_str(text):
    print(text)


encode = enc.stream_decode(print_str)
for i in ids:
    encode(i)

# enc.save_vocab("gpt4")
