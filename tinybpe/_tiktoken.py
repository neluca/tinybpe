"""TikToken model conversion utilities.

Provides functions to convert tiktoken encoding parameters into
TinyBPE-compatible model parameters.
"""

from tinybpe._model_io import BPEParam, save_bpe_model


def _bpe_pair(mergeable_ranks: dict[bytes, int], token: bytes) -> list[bytes]:
    """Decompose a token into byte pairs using mergeable ranks.

    Args:
        mergeable_ranks: Dict mapping byte sequences to their merge rank.
        token: A token as bytes.

    Returns:
        List of 2 byte sequences (the pair that produces the token).
    """
    parts = [bytes([b]) for b in token]
    max_rank = mergeable_ranks.get(token)

    while True:
        min_idx = None
        min_rank = None
        for idx, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = idx
                min_rank = rank

        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break

        if min_idx is None:
            raise ValueError(f"Failed to find merge pair for token {token!r}")
        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2:]
        )

    return parts  # type: ignore[return-value]


def get_from_tiktoken(mergeable_ranks: dict[bytes, int]) -> BPEParam:
    """Convert tiktoken ``enc._mergeable_ranks`` into TinyBPE parameters.

    Args:
        mergeable_ranks: The ``_mergeable_ranks`` dict from a tiktoken encoding.

    Returns:
        BPEParam with merges and optional byte remapping.

    Example:
        >>> import tiktoken
        >>> from tinybpe import get_from_tiktoken
        >>> enc = tiktoken.get_encoding("cl100k_base")
        >>> param = get_from_tiktoken(enc._mergeable_ranks)
    """
    merges_info: dict[int, tuple[int, int]] = {}

    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = _bpe_pair(mergeable_ranks, token)
        if len(pair) != 2:
            raise ValueError(f"Expected 2 parts for token {token!r}, got {len(pair)}")
        i_0 = mergeable_ranks[pair[0]]
        i_1 = mergeable_ranks[pair[1]]
        merges_info[rank] = (i_0, i_1)

    merges_len = len(merges_info)
    merges = [merges_info[i + 256] for i in range(merges_len)]

    # Check if byte remapping is needed
    bytes_maps = [mergeable_ranks[bytes([i])] for i in range(256)]
    for i, b in enumerate(bytes_maps):
        if i != b:
            return BPEParam(bytes_maps=bytes_maps, merges=merges)

    return BPEParam(bytes_maps=None, merges=merges)


def save_from_tiktoken(file_prefix: str, mergeable_ranks: dict[bytes, int]) -> None:
    """Convert tiktoken parameters and save as a TinyBPE model file.

    Args:
        file_prefix: Path prefix for the output .tinymodel file.
        mergeable_ranks: The ``_mergeable_ranks`` dict from a tiktoken encoding.

    Example:
        >>> import tiktoken
        >>> from tinybpe import save_from_tiktoken
        >>> enc = tiktoken.get_encoding("cl100k_base")
        >>> save_from_tiktoken("cl100k_base", enc._mergeable_ranks)
    """
    bpe_param = get_from_tiktoken(mergeable_ranks)
    save_bpe_model(file_prefix, bpe_param.merges, bytes_maps=bpe_param.bytes_maps)
