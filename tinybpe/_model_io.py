"""Model file I/O for TinyBPE.

Handles saving and loading BPE model parameters (merges and byte remapping)
to/from .tinymodel files and .vocab files.
"""

from dataclasses import dataclass
from typing import Optional

MODEL_VERSION = 1


@dataclass
class BPEParam:
    """BPE model parameters.

    Attributes:
        bytes_maps: Byte remapping table (list of 256 ints), or None for identity.
        merges: List of merge pairs (left, right) that define the BPE vocabulary.
    """
    bytes_maps: Optional[list[int]]
    merges: list[tuple[int, int]]


def save_bpe_vocab(file_prefix: str, vocab: dict[int, bytes]) -> None:
    """Save the vocabulary to a file named ``<file_prefix>.vocab``.

    The format is tab-separated: ``rank\\tbase64_encoded_bytes``,
    compatible with the tiktoken .tiktoken file format.

    Args:
        file_prefix: Path prefix for the output file.
        vocab: Dictionary mapping token IDs to byte sequences.
    """
    import base64
    vocab_file = file_prefix + ".vocab"

    with open(vocab_file, 'w', encoding="utf-8") as f:
        f.write(f"TinyBPE Vocabulary v{MODEL_VERSION}\n")
        for rank in sorted(vocab.keys()):
            encoded = base64.b64encode(vocab[rank]).decode("ascii")
            f.write(f"{encoded} {rank}\n")


def load_bpe_vocab(vocab_file: str) -> dict[int, bytes]:
    """Load a vocabulary file (tiktoken-compatible format).

    Args:
        vocab_file: Path to the .vocab file.

    Returns:
        Dictionary mapping token IDs to byte sequences.
    """
    import base64
    vocab: dict[int, bytes] = {}

    with open(vocab_file, 'r', encoding="utf-8") as f:
        header = f.readline().strip()
        if not header.startswith("TinyBPE Vocabulary"):
            raise ValueError(f"Invalid vocabulary file header: {header}")

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                raise ValueError(f"Invalid vocabulary line: {line}")
            encoded, rank_str = parts
            rank = int(rank_str)
            text_bytes = base64.b64decode(encoded)
            vocab[rank] = text_bytes

    return vocab


def save_bpe_model(
    file_prefix: str,
    merges: list[tuple[int, int]],
    bytes_maps: Optional[list[int]] = None,
) -> None:
    """Save merges and optional byte remapping to ``<file_prefix>.tinymodel``.

    Args:
        file_prefix: Path prefix for the output file.
        merges: List of merge pairs.
        bytes_maps: Optional byte remapping list of 256 integers.
    """
    bpe_file = file_prefix + ".tinymodel"

    with open(bpe_file, 'w', encoding="utf-8") as f:
        f.write(f"TinyBPE Model v{MODEL_VERSION}\n")
        if bytes_maps is not None:
            if len(bytes_maps) != 256:
                raise ValueError("bytes_maps must have exactly 256 elements")
            f.write("256\n")
            for i in bytes_maps:
                f.write(f"{i}\n")
        else:
            f.write("0\n")

        for left, right in merges:
            f.write(f"{left} {right}\n")


def load_bpe_model(model_file: str) -> BPEParam:
    """Load BPE model parameters from a .tinymodel file.

    Args:
        model_file: Path to the .tinymodel file.

    Returns:
        BPEParam with bytes_maps and merges.

    Raises:
        ValueError: If the file format is invalid or unsupported.
    """
    if not model_file.endswith(".tinymodel"):
        raise ValueError("Model file must end with .tinymodel")

    merges: list[tuple[int, int]] = []
    bytes_maps: Optional[list[int]] = []

    with open(model_file, 'r', encoding="utf-8") as f:
        magic = f.readline().strip()

        # Parse version from header
        if "v" in magic:
            version_str = magic.split("v")[-1]
            try:
                version = int(version_str)
            except ValueError:
                raise ValueError(f"Invalid model file header: {magic}")
        else:
            # Legacy format without version
            version = 0

        if version > MODEL_VERSION:
            raise ValueError(
                f"Model file version {version} is newer than "
                f"the supported version ({MODEL_VERSION})"
            )

        if not magic.startswith("TinyBPE Model"):
            raise ValueError(f"Invalid model file header: {magic}")

        # Read byte remapping
        remap_count_line = f.readline().strip()
        if remap_count_line == "256":
            for _ in range(256):
                bytes_maps.append(int(f.readline().strip()))
        elif remap_count_line == "0":
            bytes_maps = None
        else:
            raise ValueError(f"Unexpected remap count: {remap_count_line}")

        # Read merge pairs
        for line in f:
            line = line.strip()
            if not line:
                continue
            left, right = map(int, line.split())
            merges.append((left, right))

    return BPEParam(bytes_maps=bytes_maps, merges=merges)
