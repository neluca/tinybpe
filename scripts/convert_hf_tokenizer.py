#!/usr/bin/env python3
"""Convert a HuggingFace ``tokenizer.json`` to a TinyBPE ``.tbm`` model file.

Supports:
  - BPE tokenizers with ByteLevel pre-tokenizer (GPT-2 style)
  - Local ``tokenizer.json`` files
  - HuggingFace Hub model IDs (e.g. ``Qwen/Qwen2.5-0.5B``)

Usage::

    python scripts/convert_hf_tokenizer.py tokenizer.json -o output.tbm
    python scripts/convert_hf_tokenizer.py Qwen/Qwen2.5-0.5B -o models/qwen25.tbm

Dependencies: ``huggingface_hub`` (``pip install huggingface_hub``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# GPT-2 ByteLevel mapping: bytes (0-255) ↔ visible Unicode characters
# ---------------------------------------------------------------------------


def _bytes_to_unicode() -> dict[int, int]:
    """Build the standard GPT-2 byte-to-unicode mapping.

    Returns a dict mapping byte value (0-255) → unicode codepoint.
    Printable bytes (! to ~, ¡ to ÿ) keep their identity;
    non-printable bytes are mapped to codepoints 256+.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]  # copy
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))


# Build once at module level
_BYTE_TO_UNICODE = _bytes_to_unicode()
_UNICODE_TO_BYTE: dict[int, int] = {v: k for k, v in _BYTE_TO_UNICODE.items()}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


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
            "Error: huggingface_hub is required for Hub downloads.  Install it with: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    path = hf_hub_download(source, "tokenizer.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Conversion: HuggingFace tokenizer.json → TinyBPE (merges, bytes_maps)
# ---------------------------------------------------------------------------


def hf_to_tinybpe(tokenizer_json: dict) -> tuple[list[tuple[int, int]], list[int] | None]:
    """Extract BPE merge pairs and byte remapping from a HF tokenizer.json.

    Handles GPT-2-style ByteLevel pre-tokenizers where raw bytes are
    mapped to visible Unicode characters before BPE encoding.

    Parameters
    ----------
    tokenizer_json : dict
        The parsed tokenizer.json content.

    Returns
    -------
    tuple
        ``(merges, bytes_maps)``.  ``bytes_maps`` is ``None`` if no
        remapping is needed (identity).
    """
    model = tokenizer_json.get("model", {})
    model_type = model.get("type", "")

    if model_type != "BPE":
        raise ValueError(f"Unsupported model type: {model_type!r}.  Only BPE is supported.")

    vocab: dict[str, int] = model.get("vocab", {})
    if not vocab:
        raise ValueError("No 'vocab' found in tokenizer.json model section")

    merges_raw: list[str] = model.get("merges", [])
    if not merges_raw:
        raise ValueError("No 'merges' found in tokenizer.json model section")

    # ------------------------------------------------------------------
    # Byte remapping for ByteLevel pre-tokenizers.
    #
    # ByteLevel maps raw bytes (0-255) to visible Unicode characters.
    # Most tokenizers use the GPT-2 _bytes_to_unicode mapping, but some
    # (like DeepSeek) use a slightly different set.  We auto-detect the
    # mapping from the vocab by finding which single-char tokens
    # represent which bytes.
    #
    # Our bytes_maps[i] = token ID that represents raw byte i.
    # Input byte i is replaced by bytes_maps[i] before BPE encoding.
    # ------------------------------------------------------------------
    bytes_maps: list[int] | None = None
    is_bytelevel = _detect_bytelevel(tokenizer_json)

    if is_bytelevel:
        # Auto-detect the byte-to-unicode mapping from the vocab
        detected_mapping = _detect_byte_mapping(vocab)
        if detected_mapping is None:
            raise ValueError("ByteLevel tokenizer has incomplete byte mapping in vocab")
        bytes_maps = detected_mapping

        # Check if remapping is actually identity
        if all(bytes_maps[i] == i for i in range(256)):
            bytes_maps = None

    # ------------------------------------------------------------------
    # Build merge pairs from merge strings.
    #
    # Merge strings look like "Ġ t" (space + t) or "i n" (i + n).
    # We split on space and look up each token in the vocab.
    # ------------------------------------------------------------------
    merges: list[tuple[int, int]] = []
    for merge_str in merges_raw:
        parts = merge_str.split(" ")
        if len(parts) != 2:
            continue  # skip malformed entries
        a, b = parts
        left = vocab.get(a)
        right = vocab.get(b)
        if left is None:
            print(f"  Warning: token {a!r} not in vocab, skipping merge {merge_str!r}", file=sys.stderr)
            continue
        if right is None:
            print(f"  Warning: token {b!r} not in vocab, skipping merge {merge_str!r}", file=sys.stderr)
            continue
        merges.append((left, right))

    return merges, bytes_maps


def _detect_byte_mapping(vocab: dict[str, int]) -> list[int] | None:
    """Auto-detect byte-to-token-ID mapping from a ByteLevel vocab.

    ByteLevel maps raw bytes (0-255) to visible Unicode characters.
    Printable ASCII (33-126) and Latin-1 supplement (161-172, 174-255)
    typically map to themselves.  Non-printable bytes map to codepoints
    256+.

    Returns a list of 256 token IDs, or None if not enough single-char
    tokens are found.
    """
    # Collect all single-char tokens and their codepoints
    char_to_id: dict[str, int] = {}
    for tok, tid in vocab.items():
        if len(tok) == 1:
            char_to_id[tok] = tid

    # Strategy: use the standard GPT-2 mapping as a starting point,
    # but fall back to heuristic matching for tokenizers that deviate.
    bytes_maps = [-1] * 256

    # First pass: try GPT-2 mapping
    for byte_val in range(256):
        unicode_codepoint = _BYTE_TO_UNICODE[byte_val]
        unicode_char = chr(unicode_codepoint)
        if unicode_char in char_to_id:
            bytes_maps[byte_val] = char_to_id[unicode_char]

    # Second pass: for any missing bytes, try to find them among unused
    # single-char tokens. Some tokenizers (DeepSeek) omit rare byte
    # mappings from the vocab because those bytes never appear in valid
    # UTF-8 (0xC0-0xC1 overlong encoding, 0xF5-0xFF exceed RFC 3629).
    still_missing = [b for b in range(256) if bytes_maps[b] < 0]
    if still_missing:
        # Collect all single-char tokens not yet assigned
        assigned_ids = {bytes_maps[b] for b in range(256) if bytes_maps[b] >= 0}
        unassigned = [(ord(c), tid) for c, tid in char_to_id.items() if tid not in assigned_ids]
        unassigned.sort()

        for byte_val in still_missing:
            expected_uc = _BYTE_TO_UNICODE[byte_val]
            # Find the closest unassigned token by codepoint
            best = None
            best_dist = 999999
            for uc, tid in unassigned:
                dist = abs(uc - expected_uc)
                if dist < best_dist:
                    best = (uc, tid)
                    best_dist = dist
            if best is not None:
                bytes_maps[byte_val] = best[1]
                unassigned.remove(best)

    # Third pass: any bytes still unmapped are unreachable in valid UTF-8.
    # 0xC0-0xC1 (overlong encoding) and 0xF5-0xFF (exceed RFC 3629)
    # never appear in valid UTF-8 text.  Assign them to unused token IDs
    # in the 0-255 range so bytes_maps remains a bijection (invertible).
    still_missing = [b for b in range(256) if bytes_maps[b] < 0]
    if still_missing:
        for byte_val in still_missing:
            if byte_val <= 0x7F or (0xC2 <= byte_val <= 0xF4):
                # Valid UTF-8 lead byte is missing — real problem
                return None

        # Find unused token IDs in 0-255 (standard base-byte range)
        assigned_ids = {bytes_maps[b] for b in range(256) if bytes_maps[b] >= 0}
        free_ids = [i for i in range(256) if i not in assigned_ids]
        # Sort to assign lowest free IDs to lowest missing bytes (deterministic)
        free_ids.sort()
        missing_sorted = sorted(still_missing)

        if len(free_ids) < len(missing_sorted):
            return None  # should not happen for valid tokenizers

        for byte_val, free_id in zip(missing_sorted, free_ids):
            bytes_maps[byte_val] = free_id

    return bytes_maps


def _detect_bytelevel(tokenizer_json: dict) -> bool:
    """Check if the tokenizer uses ByteLevel pre-tokenization."""
    pre_tokenizer = tokenizer_json.get("pre_tokenizer", {})
    if not pre_tokenizer:
        # Check decoder as fallback
        decoder = tokenizer_json.get("decoder", {})
        return decoder.get("type") == "ByteLevel"

    # Handle nested Sequence pre_tokenizer
    pretoks = pre_tokenizer.get("pretokenizers", [pre_tokenizer])
    for pt in pretoks:
        if pt.get("type") == "ByteLevel":
            return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a HuggingFace tokenizer.json to a TinyBPE .tbm file.")
    parser.add_argument(
        "source",
        help="Path to tokenizer.json or HuggingFace model ID",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output .tbm file path",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.source}")
    tokenizer_json = load_tokenizer_json(args.source)

    model_type = tokenizer_json.get("model", {}).get("type", "unknown")
    vocab_size = len(tokenizer_json.get("model", {}).get("vocab", {}))
    print(f"  Type: {model_type}")
    print(f"  Vocab: {vocab_size}")

    is_bytelevel = _detect_bytelevel(tokenizer_json)
    print(f"  ByteLevel: {'yes' if is_bytelevel else 'no'}")

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
