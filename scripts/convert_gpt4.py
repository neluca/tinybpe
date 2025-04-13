import tiktoken
from tools_tiktoken import bpe_mergeable_ranks_save

enc = tiktoken.get_encoding("cl100k_base")

bpe_mergeable_ranks_save("cl100k_base", enc._mergeable_ranks)
print(enc._special_tokens)
