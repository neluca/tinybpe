import tiktoken
from tinybpe import Tokenizer, get_from_tiktoken

tik_tokenizer = tiktoken.get_encoding("cl100k_base")
model_param = get_from_tiktoken(tik_tokenizer._mergeable_ranks)
tiny_tokenizer = Tokenizer(model_param)

text = "👋 Hello, this is an example. 你好，这是一个例子。😁"
tik_ids = tik_tokenizer.encode(text)
tiny_ids = tiny_tokenizer.encode(text)
assert tik_ids == tiny_ids
print("tiktoken: ", tik_ids)
print("tinybpe: ", tiny_ids)

tik_text = tik_tokenizer.decode(tiny_ids)
tiny_text = tiny_tokenizer.decode(tik_ids)
assert tik_text == tiny_text

print("tiktoken: ", tik_text)
print("tinybpe: ", tiny_text)

# output:
# tiktoken:  [9468, 239, 233, 22691, 11, 420, 374, 459, 3187, 13, 220, 57668, 53901, 3922, 44388, 21043, 48044, 27452, 45829, 1811, 76460, 223]
# tinybpe:  [9468, 239, 233, 22691, 11, 420, 374, 459, 3187, 13, 220, 57668, 53901, 3922, 44388, 21043, 48044, 27452, 45829, 1811, 76460, 223]
# tiktoken:  👋 Hello, this is an example. 你好，这是一个例子。😁
# tinybpe:  👋 Hello, this is an example. 你好，这是一个例子。😁
