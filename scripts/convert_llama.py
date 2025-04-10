from tools import bpe_load_remaps

maps = bpe_load_remaps("cl100k_base.map")
print(len(maps))
print(maps)
