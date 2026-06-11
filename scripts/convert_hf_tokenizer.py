#!/usr/bin/env python3
"""Convert a HuggingFace ``tokenizer.json`` to a TinyBPE ``.tbm`` model file.

Supports:
  - Local ``tokenizer.json`` files
  - HuggingFace Hub model IDs (e.g. ``meta-llama/Meta-Llama-3-8B``)

Usage::

    python scripts/convert_hf_tokenizer.py path/to/tokenizer.json -o output.tbm
    python scripts/convert_hf_tokenizer.py meta-llama/Meta-Llama-3-8B -o models/llama3.tbm

Dependencies: ``huggingface_hub`` (install with ``pip install huggingface_hub``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_tokenizer_json(source: str) -> dict:
    """Load a tokenizer.json from a local path or HuggingFace Hub model ID."""
    if Path(source).exists():
        with open(source, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    # Try HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Error: huggingface_hub is required for Hub downloads.  "
            "Install it with: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    path = hf_hub_download(source, "tokenizer.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def hf_to_tinybpe(tokenizer_json: dict) -> tuple[list[tuple[int, int]], list[int] | None]:
    """Extract BPE merge pairs from a HuggingFace tokenizer.json.

    Parameters
    ----------
    tokenizer_json : dict
        The parsed tokenizer.json content.

    Returns
    -------
    tuple
        ``(merges, bytes_maps)``.
    """
    model = tokenizer_json.get("model", {})
    vocab: dict[str, int] = model.get("vocab", {})

    if not vocab:
        raise ValueError("No 'vocab' found in tokenizer.json model section")

    # Build reverse mapping: id → token string
    id_to_token: dict[int, str] = {v: k for k, v in vocab.items()}

    # Extract merge lines from the model
    merges_raw = model.get("merges", [])
    if not merges_raw:
        raise ValueError("No 'merges' found in tokenizer.json model section")

    # Parse merge strings like "h e" → individual tokens
    merges: list[tuple[int, int]] = []
    bytes_maps: list[int] | None = None

    # First pass: determine if byte-level tokens are remapped
    # Standard BPE uses single bytes as base; some tokenizers remap them
    base_map: dict[str, int] = {}
    for token_str, tid in vocab.items():
        token_bytes = bytes(token_str, "utf-8") if isinstance(token_str, str) else token_str
        if len(token_bytes) == 1:
            base_map[token_str] = tid

    # Check if remapping is needed (base byte tokens not at IDs 0-255)
    needs_remap = False
    if len(base_map) >= 256:
        # Sort by byte value
        sorted_bytes = sorted(base_map.items(), key=lambda x: ord(x[0]) if isinstance(x[0], str) else x[0])
        if len(sorted_bytes) == 256:
            for i, (_, tid) in enumerate(sorted_bytes):
                if tid != i:
                    needs_remap = True
                    break

    if needs_remap:
        bytes_maps = [0] * 256
        # Build the remap from byte value → token ID
        for i in range(256):
            byte_char = chr(i)
            if byte_char in vocab:
                bytes_maps[i] = vocab[byte_char]
            else:
                bytes_maps[i] = i  # identity fallback

    # Build merges from merge strings
    for merge_str in merges_raw:
        if isinstance(merge_str, str):
            parts = merge_str.split(" ")
            if len(parts) == 2:
                a, b = parts
                left = vocab.get(a)
                right = vocab.get(b)
                if left is not None and right is not None:
                    merges.append((left, right))

    return merges, bytes_maps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace tokenizer.json to a TinyBPE .tbm file."
    )
    parser.add_argument(
        "source",
        help="Path to tokenizer.json or HuggingFace model ID",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output .tbm file path",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.source}")
    tokenizer_json = load_tokenizer_json(args.source)

    print("Converting to TinyBPE format...")
    merges, bytes_maps = hf_to_tinybpe(tokenizer_json)

    print(f"  Merges: {len(merges)}")
    if bytes_maps is not None:
        print("  Byte remapping: yes")
    else:
        print("  Byte remapping: no")

    from tinybpe._model_io import save_model
    save_model(args.output, merges, bytes_maps)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
