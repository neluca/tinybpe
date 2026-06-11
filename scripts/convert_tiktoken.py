#!/usr/bin/env python3
"""Convert a tiktoken encoding to a TinyBPE ``.tbm`` model file.

Usage::

    python scripts/convert_tiktoken.py cl100k_base -o models/cl100k_base.tbm
    python scripts/convert_tiktoken.py o200k_base -o models/o200k_base.tbm
    python scripts/convert_tiktoken.py p50k_base -o models/p50k_base.tbm
    python scripts/convert_tiktoken.py r50k_base -o models/r50k_base.tbm

Dependencies: ``tiktoken`` (install with ``pip install tiktoken``).
"""

from __future__ import annotations

import argparse
import sys


def _decompose_token(
    mergeable_ranks: dict[bytes, int], token: bytes
) -> list[bytes]:
    """Decompose a token into its constituent byte pair.

    Uses the merge ranks to greedily find the lowest-rank pair that
    produces the given token.
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
            + parts[min_idx + 2 :]
        )

    return parts


def tiktoken_to_tinybpe(
    mergeable_ranks: dict[bytes, int],
) -> tuple[list[tuple[int, int]], list[int] | None]:
    """Convert tiktoken ``_mergeable_ranks`` to TinyBPE parameters.

    Parameters
    ----------
    mergeable_ranks : dict[bytes, int]
        The ``_mergeable_ranks`` dict from a tiktoken encoding.

    Returns
    -------
    tuple
        ``(merges, bytes_maps)`` where ``bytes_maps`` may be ``None``
        if no byte remapping is needed.
    """
    merges_info: dict[int, tuple[int, int]] = {}

    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = _decompose_token(mergeable_ranks, token)
        if len(pair) != 2:
            raise ValueError(
                f"Expected 2 parts for token {token!r}, got {len(pair)}"
            )
        i_0 = mergeable_ranks[pair[0]]
        i_1 = mergeable_ranks[pair[1]]
        merges_info[rank] = (i_0, i_1)

    merges_len = len(merges_info)
    merges = [merges_info[i + 256] for i in range(merges_len)]

    # Detect byte remapping
    bytes_maps = [mergeable_ranks[bytes([i])] for i in range(256)]
    for i, b in enumerate(bytes_maps):
        if i != b:
            return merges, bytes_maps

    return merges, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a tiktoken encoding to a TinyBPE .tbm model file."
    )
    parser.add_argument(
        "encoding",
        help="Tiktoken encoding name (e.g. cl100k_base, o200k_base, p50k_base, r50k_base)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output .tbm file path",
    )
    args = parser.parse_args()

    try:
        import tiktoken
    except ImportError:
        print(
            "Error: tiktoken is required.  Install it with: pip install tiktoken",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading tiktoken encoding: {args.encoding}")
    enc = tiktoken.get_encoding(args.encoding)

    print(f"  Vocab size: {enc.n_vocab}")
    print("Converting to TinyBPE format...")
    merges, bytes_maps = tiktoken_to_tinybpe(enc._mergeable_ranks)

    print(f"  Merges: {len(merges)}")
    if bytes_maps is not None:
        print("  Byte remapping: yes")
    else:
        print("  Byte remapping: no")

    # Save via tinybpe's model I/O
    from tinybpe._model_io import save_model
    save_model(args.output, merges, bytes_maps)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
