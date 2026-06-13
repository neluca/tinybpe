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


def _decompose_token(mergeable_ranks: dict[bytes, int], token: bytes) -> list[bytes]:
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

        parts = [*parts[:min_idx], parts[min_idx] + parts[min_idx + 1], *parts[min_idx + 2:]]

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
    # Build mapping from tiktoken rank → our normalized ID.
    # Single bytes: our ID = byte value (0-255).
    # Multi-byte tokens: our ID = 256 + position in sorted merge order.
    # Byte remapping is applied separately — bytes_maps[i] = tiktoken rank of byte i.
    bytes_maps = [mergeable_ranks[bytes([i])] for i in range(256)]
    has_remap = any(i != b for i, b in enumerate(bytes_maps))

    # Build tiktoken_rank → our_normalized_id mapping.
    #
    # In TinyBPE, bytes_maps remaps input bytes BEFORE encoding.
    # So the C tokenizer sees the POST-REMAP byte values.
    # Merge pairs must reference these post-remap IDs.
    #
    # For single bytes: our ID = the remapped value (tiktoken rank).
    #   E.g. if bytes_maps[32]=220, then space is token 220 in the C tokenizer.
    # For multi-byte tokens: our ID = 256 + position in sorted merge order.
    rank_to_id: dict[int, int] = {}

    # Single bytes: keep the remapped (tiktoken rank) as our token ID
    for byte_val in range(256):
        tiktok_rank = bytes_maps[byte_val]
        rank_to_id[tiktok_rank] = tiktok_rank

    # Multi-byte tokens: collect and sort by rank
    merges_info: dict[int, tuple[int, int]] = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = _decompose_token(mergeable_ranks, token)
        if len(pair) != 2:
            raise ValueError(f"Expected 2 parts for token {token!r}, got {len(pair)}")
        # Use tiktoken's internal rank for the pair components
        i_0 = mergeable_ranks[pair[0]]
        i_1 = mergeable_ranks[pair[1]]
        merges_info[rank] = (i_0, i_1)

    # Sort multi-byte token ranks and assign sequential IDs
    sorted_ranks = sorted(merges_info.keys())
    for new_id, tiktok_rank in enumerate(sorted_ranks, start=256):
        rank_to_id[tiktok_rank] = new_id

    # Now translate merge pairs from tiktoken ranks → our normalized IDs
    merges: list[tuple[int, int]] = []
    for tiktok_rank in sorted_ranks:
        left_tiktok, right_tiktok = merges_info[tiktok_rank]
        # Guard against missing mappings (shouldn't happen for valid encodings)
        left_id = rank_to_id.get(left_tiktok)
        right_id = rank_to_id.get(right_tiktok)
        if left_id is None or right_id is None:
            raise ValueError(f"Missing rank mapping: left={left_tiktok}, right={right_tiktok}")
        merges.append((left_id, right_id))

    if has_remap:
        return merges, bytes_maps
    return merges, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a tiktoken encoding to a TinyBPE .tbm model file.")
    parser.add_argument(
        "encoding",
        help="Tiktoken encoding name (e.g. cl100k_base, o200k_base, p50k_base, r50k_base)",
    )
    parser.add_argument(
        "-o",
        "--output",
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
