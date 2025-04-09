# import tiktoken
# from tinybpe import bpe
from tools import bpe_save_merges, bpe_save_remaps


def _bpe_pair(mergeable_ranks: dict[bytes, int], token: bytes) -> list[bytes, bytes]:
    parts = [bytes([b]) for b in token]
    max_rank = mergeable_ranks[token]

    while True:
        min_idx, min_rank = (None, None)
        for idx, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx, min_rank = (idx, rank)
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def bpe_get_merges_and_remaps(mergeable_ranks: dict[bytes, int]) -> tuple[list, list | None]:
    merges_info = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = _bpe_pair(mergeable_ranks, token)
        assert len(pair) == 2
        i_0 = mergeable_ranks[pair[0]]
        i_1 = mergeable_ranks[pair[1]]
        merges_info[rank] = (i_0, i_1)

    merges_len = len(merges_info)
    merges = [merges_info[i + 256] for i in range(merges_len)]
    remaps = [mergeable_ranks[bytes([i])] for i in range(256)]
    for i, b in enumerate(remaps):
        if i != b:
            return merges, remaps

    return merges, None


def bpe_mergeable_ranks_save(file_prefix: str, mergeable_ranks: dict[bytes, int]):
    merges, remaps = bpe_get_merges_and_remaps(mergeable_ranks)
    bpe_save_merges(file_prefix, merges)
    if remaps is not None:
        bpe_save_remaps(file_prefix, remaps)

# enc = tiktoken.get_encoding("cl100k_base")
# mergeable_ranks = enc._mergeable_ranks
#
# merges_, remap_ = bpe_get_merges_and_remap(mergeable_ranks)
#
# bytes_remap = bpe.BytesRemap(remap_)

# print(merges_)
# print(remap_)
#
# s = b"abcdefg 1234567"
# print(bytes_remap(s))
