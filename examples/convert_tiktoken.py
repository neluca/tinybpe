import tiktoken
from tinybpe import save_from_tiktoken

enc = tiktoken.get_encoding("cl100k_base")
save_from_tiktoken("cl100k_base", enc._mergeable_ranks)
print(enc._special_tokens)
