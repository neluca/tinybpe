#!/usr/bin/env python3
"""Convert MiniCPM SentencePiece-BPE tokenizer to TinyBPE .tbm format.

Converts character-level BPE to byte-level BPE with proper dependency
ordering for TinyBPE's greedy scan engine.

**Important**: Token IDs differ from MiniCPM's originals. Decoded text is
identical. Not suitable for model inference requiring specific token IDs.

Usage::

    python scripts/convert_minicpm.py -o models/minicpm.tbm
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# GPT-2 ByteLevel mapping
# ---------------------------------------------------------------------------


def _bytes_to_unicode() -> dict[int, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))


_BYTE_TO_UNICODE = _bytes_to_unicode()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_tokenizer_json(source: str) -> dict:
    if Path(source).exists():
        with open(source, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    try:
        import requests

        url = f"https://modelscope.cn/api/v1/models/{source}/repo?Revision=master&FilePath=tokenizer.json"
        r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            return r.json()  # type: ignore[no-any-return]
    except Exception:
        pass

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(source, "tokenizer.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]
    except ImportError:
        print("Error: huggingface_hub required.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Trie-based byte-level building
# ---------------------------------------------------------------------------


class _TrieNode:
    """Node in a byte prefix trie."""

    __slots__ = ("byte_val", "children", "token_id")

    def __init__(self, byte_val: int = -1):
        self.children: dict[int, _TrieNode] = {}
        self.token_id: int | None = None  # set for leaf tokens
        self.byte_val = byte_val


def _build_byte_trie(byte_sequences: list[tuple[int, bytes]]) -> _TrieNode:
    """Build a prefix trie from (old_id, byte_sequence) pairs.

    Returns the root node. Each leaf stores the old token ID.
    """
    root = _TrieNode()
    for old_id, seq in byte_sequences:
        node = root
        for b in seq:
            if b not in node.children:
                node.children[b] = _TrieNode(b)
            node = node.children[b]
        node.token_id = old_id  # only set for the final byte
    return root


def _trie_to_merges(root: _TrieNode) -> tuple[list[tuple[int, int]], dict[int, int], dict[int, bytes]]:
    """Convert a byte prefix trie to BPE merges.

    Traverses the trie breadth-first. For each edge (parent_node → child_node
    via byte B), creates a merge (parent_token_id, B) → new_token_id.

    Returns:
        merges: list of (parent_id, byte_val) in dependency order
        old_to_new: mapping from old token ID → new ID
        new_id_to_bytes: mapping from new ID → byte sequence
    """
    merges: list[tuple[int, int]] = []
    old_to_new: dict[int, int] = {}
    new_id_to_bytes: dict[int, bytes] = {}

    # Base bytes: 0-255
    for b in range(256):
        new_id_to_bytes[b] = bytes([b])

    next_id = 256

    # BFS queue: (node, token_id_for_node, node_bytes)
    queue: deque[tuple[_TrieNode, int, bytes]] = deque()

    # Process root's children (1-byte prefixes).
    # Each child of root represents a prefix of length 1, which is already
    # a token (byte value 0-255).  For each grandchild, create a merge
    # that appends the grandchild's edge byte to the 1-byte prefix.
    for edge_byte, child in sorted(root.children.items()):
        for child_edge, grandchild in sorted(child.children.items()):
            my_id = next_id
            next_id += 1
            merges.append((edge_byte, child_edge))
            my_bytes = bytes([edge_byte, child_edge])
            new_id_to_bytes[my_id] = my_bytes
            if grandchild.token_id is not None:
                old_to_new[grandchild.token_id] = my_id
            queue.append((grandchild, my_id, my_bytes))

    # BFS for deeper levels
    while queue:
        node, parent_token_id, prefix_bytes = queue.popleft()

        for edge_byte, child in sorted(node.children.items()):
            my_id = next_id
            next_id += 1
            merges.append((parent_token_id, edge_byte))
            my_bytes = prefix_bytes + bytes([edge_byte])
            new_id_to_bytes[my_id] = my_bytes
            if child.token_id is not None:
                old_to_new[child.token_id] = my_id
            queue.append((child, my_id, my_bytes))

    return merges, old_to_new, new_id_to_bytes


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def minicpm_to_tinybpe(tokenizer_json: dict) -> tuple[list[tuple[int, int]], dict[int, bytes]]:
    """Convert MiniCPM tokenizer to byte-level BPE."""
    model = tokenizer_json.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"Unsupported model type: {model.get('type')!r}")

    old_vocab: dict[str, int] = model.get("vocab", {})
    old_merges_raw: list[str] = model.get("merges", [])

    print(f"  Old vocab: {len(old_vocab)} tokens, {len(old_merges_raw)} merges")

    # ---- build lookup tables ----
    old_id_to_bytes: dict[int, bytes] = {}
    old_id_to_str: dict[int, str] = {}
    for tok_str, tok_id in old_vocab.items():
        old_id_to_bytes[tok_id] = tok_str.encode("utf-8")
        old_id_to_str[tok_id] = tok_str

    # ---- parse merges ----
    parsed_merges: list[tuple[int, int, int | None, str]] = []
    for merge_str in old_merges_raw:
        parts = merge_str.split(" ")
        if len(parts) != 2:
            continue
        left = old_vocab.get(parts[0])
        right = old_vocab.get(parts[1])
        if left is None or right is None:
            continue
        merged_str = parts[0] + parts[1]
        result = old_vocab.get(merged_str)
        parsed_merges.append((left, right, result, merged_str))

    print(f"  Parsed merges: {len(parsed_merges)}")

    # ---- identify tokens ----
    merge_result_ids: set[int] = {r for _, _, r, _ in parsed_merges if r is not None}
    base_token_ids: set[int] = set(old_vocab.values()) - merge_result_ids
    print(f"  Base tokens: {len(base_token_ids)}, merge-result: {len(merge_result_ids)}")

    # ---- new_id mapping ----
    new_id: dict[int, int] = {}
    for byte_val in range(256):
        uc = _BYTE_TO_UNICODE[byte_val]
        ch = chr(uc)
        if ch in old_vocab:
            old_id = old_vocab[ch]
            new_id[old_id] = byte_val
        else:
            raise ValueError(f"Byte {byte_val} ({ch!r}) not in vocab")

    # ---- collect multi-byte base tokens for trie building ----
    trie_entries: list[tuple[int, bytes]] = []  # (old_id, byte_seq)
    for old_id in base_token_ids:
        tb = old_id_to_bytes.get(old_id)
        if tb is None:
            continue
        if len(tb) == 1:
            # Single byte — already mapped
            continue
        trie_entries.append((old_id, tb))

    print(f"  Multi-byte base tokens for trie: {len(trie_entries)}")

    # ---- build shared prefix trie and convert to merges ----
    root = _build_byte_trie(trie_entries)
    trie_merges, trie_old_to_new, new_id_to_bytes = _trie_to_merges(root)

    # Merge the trie mapping into new_id
    new_id.update(trie_old_to_new)

    print(f"  Trie merges: {len(trie_merges)}")
    print(f"  Mapped base tokens: {len(trie_old_to_new)}")

    # ---- build result: start with trie merges ----
    all_merges: list[tuple[int, int]] = list(trie_merges)
    seen_merge_pairs: set[tuple[int, int]] = set(trie_merges)

    # ---- track which byte sequences we've already created ----
    # This prevents creating duplicate merges for the same byte sequence
    # Build the inverse mapping: bytes → new_id
    created_bytes: dict[bytes, int] = {v: k for k, v in new_id_to_bytes.items()}

    # ---- process MiniCPM merges in dependency order ----
    remaining = list(parsed_merges)
    processed = 0
    skipped_duplicates = 0
    max_iterations = len(remaining) * 2
    iteration = 0
    next_id = 256 + len(trie_merges)

    while remaining and iteration < max_iterations:
        iteration += 1
        next_remaining: list[tuple[int, int, int | None, str]] = []
        progress = False

        for left_old, right_old, result_old, merged_str in remaining:
            if left_old not in new_id or right_old not in new_id:
                next_remaining.append((left_old, right_old, result_old, merged_str))
                continue

            new_left = new_id[left_old]
            new_right = new_id[right_old]

            # Compute the result byte sequence
            merged_bytes = (
                old_id_to_bytes[result_old]
                if result_old is not None
                else new_id_to_bytes[new_left] + new_id_to_bytes[new_right]
            )

            # Check if this byte sequence already has a token
            if merged_bytes in created_bytes:
                # Already exists — reuse existing token ID
                existing_new_id = created_bytes[merged_bytes]
                if result_old is not None:
                    new_id[result_old] = existing_new_id
                skipped_duplicates += 1
                processed += 1
                progress = True
                continue

            # Check if the merge pair already exists
            if (new_left, new_right) in seen_merge_pairs:
                # Same merge pair would create a different byte sequence?
                # This shouldn't happen in a valid BPE model
                skipped_duplicates += 1
                processed += 1
                progress = True
                continue

            result_new_id = next_id
            next_id += 1
            all_merges.append((new_left, new_right))
            seen_merge_pairs.add((new_left, new_right))
            new_id_to_bytes[result_new_id] = merged_bytes
            created_bytes[merged_bytes] = result_new_id
            if result_old is not None:
                new_id[result_old] = result_new_id

            processed += 1
            progress = True

        remaining = next_remaining

        if not progress and remaining:
            # Try to resolve stuck merges by looking for alternative
            # paths to build the missing tokens
            for left_old, right_old, _result_old, _merged_str in remaining:
                if left_old not in new_id:
                    tb = old_id_to_bytes.get(left_old)
                    if tb and tb in created_bytes:
                        new_id[left_old] = created_bytes[tb]
                if right_old not in new_id:
                    tb = old_id_to_bytes.get(right_old)
                    if tb and tb in created_bytes:
                        new_id[right_old] = created_bytes[tb]

    if remaining:
        print(f"  Warning: {len(remaining)} merges could not be processed")
    print(f"  Processed merges: {processed}/{len(parsed_merges)}")
    if skipped_duplicates:
        print(f"  Skipped duplicates: {skipped_duplicates}")

    # Validate
    for idx, (left, right) in enumerate(all_merges):
        max_valid = 256 + idx
        assert left < max_valid, f"merge {idx}: left {left} >= {max_valid}"
        assert right < max_valid, f"merge {idx}: right {right} >= {max_valid}"

    # Verify no duplicate merge pairs
    assert len(set(all_merges)) == len(all_merges), "Duplicate merge pairs found!"

    print(f"  Total merges: {len(all_merges)}, vocab: {256 + len(all_merges)}")
    return all_merges, new_id_to_bytes


# ---------------------------------------------------------------------------
# .tbm I/O
# ---------------------------------------------------------------------------

MODEL_VERSION = 1


def _save_model(path: str, merges: list[tuple[int, int]]) -> None:
    if Path(path).suffix != ".tbm":
        path += ".tbm"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TinyBPE Model v{MODEL_VERSION}\n")
        f.write("0\n")  # no byte remapping
        for left, right in merges:
            f.write(f"{left} {right}\n")


def _save_vocab(path: str, vocab: dict[int, bytes]) -> None:
    import base64

    if not path.endswith(".vocab"):
        path += ".vocab"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TinyBPE Vocabulary v{MODEL_VERSION}\n")
        for rank in sorted(vocab.keys()):
            encoded = base64.b64encode(vocab[rank]).decode("ascii")
            f.write(f"{encoded} {rank}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MiniCPM tokenizer to .tbm")
    parser.add_argument(
        "source", nargs="?", default="OpenBMB/MiniCPM-2B-sft-bf16", help="Path or ModelScope/HF model ID"
    )
    parser.add_argument("-o", "--output", required=True, help="Output .tbm path")
    parser.add_argument("--save-vocab", default=None, help="Also save .vocab file")
    args = parser.parse_args()

    print(f"Loading: {args.source}")
    tokenizer_json = load_tokenizer_json(args.source)

    mt = tokenizer_json.get("model", {}).get("type", "unknown")
    vs = len(tokenizer_json.get("model", {}).get("vocab", {}))
    print(f"  Type: {mt}, Vocab: {vs}")

    print("Converting...")
    merges, id_to_bytes = minicpm_to_tinybpe(tokenizer_json)

    _save_model(args.output, merges)
    print(f"Saved: {args.output}")

    if args.save_vocab:
        _save_vocab(args.save_vocab, id_to_bytes)
        print(f"Saved vocab: {args.save_vocab}")

    size_kb = Path(args.output).stat().st_size / 1024
    print(f"  Size: {size_kb:.1f} KB, Vocab: {len(id_to_bytes)}")


if __name__ == "__main__":
    main()
