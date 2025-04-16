from dataclasses import dataclass
from typing import Optional


@dataclass
class BPEParam:
    bytes_maps: Optional[list[int]]  # Byte Remapping
    merges: list[tuple[int, int]]  # It can be also used to generate a vocabulary


def save_bpe_vocab(file_prefix: str, vocab: dict[int, bytes]):
    """ Save the vocab to a vocabulary file named <file_prefix>.vocab . """
    vocab_file = file_prefix + ".vocab"

    with open(vocab_file, 'w') as file:
        file.write("TinyBPE Vocabulary\n")
        file.write(f"n_vocab: {len(vocab)}\n")

        for rank, text_bytes in vocab.items():
            file.write(f"{rank}: {text_bytes}\n")  # type: ignore


def save_bpe_model(file_prefix: str, merges: list[tuple[int, int]], bytes_maps: Optional[list[int]] = None):
    """ Save the merges and bytes_maps parameters to a model file named <file_prefix>.tinymodel . """
    bpe_file = file_prefix + ".tinymodel"

    with open(bpe_file, 'w') as file:
        file.write("TinyBPE Model\n")
        if bytes_maps is not None:
            assert len(bytes_maps) == 256
            file.write("256\n")
            for i in bytes_maps:
                file.write(f"{i}\n")
        else:
            file.write("0\n")

        for p1, p2 in merges:
            file.write(f"{p1} {p2}\n")


def load_bpe_model(model_file: str) -> BPEParam:
    assert model_file.endswith(".tinymodel")
    merges = []
    bytes_maps = []

    with open(model_file, 'r', encoding="utf-8") as file:
        magic = file.readline().strip()
        assert magic == "TinyBPE Model"
        if int(file.readline().strip()) != 0:
            for _ in range(256):
                bytes_maps.append(int(file.readline().strip()))
        else:
            bytes_maps = None  # type: ignore

        for line in file:
            p1, p2 = map(int, line.split())
            merges.append((p1, p2))
    return BPEParam(bytes_maps=bytes_maps, merges=merges)


def _bpe_pair(mergeable_ranks: dict[bytes, int], token: bytes) -> list[bytes, bytes]:  # type: ignore
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


def get_from_tiktoken(mergeable_ranks: dict[bytes, int]) -> BPEParam:
    """ Convert `enc._mergeable_ranks` from `enc = tiktoken.get_encoding(<name>)` into tinybpe parameters. """
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
    bytes_maps = [mergeable_ranks[bytes([i])] for i in range(256)]
    for i, b in enumerate(bytes_maps):
        if i != b:
            return BPEParam(bytes_maps=bytes_maps, merges=merges)
    return BPEParam(bytes_maps=None, merges=merges)


def save_from_tiktoken(file_prefix: str, mergeable_ranks: dict[bytes, int]):
    """ Convert `enc._mergeable_ranks` from `enc = tiktoken.get_encoding(<name>)` into tinybpe model. """
    bpe_param = get_from_tiktoken(mergeable_ranks)
    save_bpe_model(file_prefix, bpe_param.merges, bytes_maps=bpe_param.bytes_maps)
