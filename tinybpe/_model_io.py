"""Model file I/O for TinyBPE.

Provides functions to save and load BPE model parameters to/from
``.tbm`` (TinyBPE Model) and ``.vocab`` files.

.tbm format (text)::

    TinyBPE Model v1
    <remap_flag>        # "0" (no remap) or "256" (has remap)
    [256 lines of remap values if remap_flag=256]
    <left> <right>      # one merge pair per line, left and right are ints

.vocab format (text)::

    TinyBPE Vocabulary v1
    <base64_token_bytes> <rank>
    ...
"""

from __future__ import annotations

from pathlib import Path

MODEL_VERSION = 1


# ---------------------------------------------------------------------------
# .tbm — model file
# ---------------------------------------------------------------------------


def save_model(
    path: str,
    merges: list[tuple[int, int]],
    bytes_maps: list[int] | None = None,
) -> None:
    """Save merges and optional byte remapping to a ``.tbm`` file.

    If ``path`` does not end with ``.tbm``, the extension is appended.

    Parameters
    ----------
    path : str
        Output file path (``.tbm`` appended if missing).
    merges : list[tuple[int, int]]
        BPE merge pairs.
    bytes_maps : list[int] or None
        Optional byte remapping table of exactly 256 integers.
    """
    if Path(path).suffix != ".tbm":
        path += ".tbm"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TinyBPE Model v{MODEL_VERSION}\n")

        if bytes_maps is not None:
            if len(bytes_maps) != 256:
                raise ValueError("bytes_maps must have exactly 256 elements")
            f.write("256\n")
            for v in bytes_maps:
                f.write(f"{v}\n")
        else:
            f.write("0\n")

        for left, right in merges:
            f.write(f"{left} {right}\n")


def load_model(path: str) -> tuple[list[tuple[int, int]], list[int] | None]:
    """Load BPE model parameters from a ``.tbm`` file.

    Parameters
    ----------
    path : str
        Path to the ``.tbm`` file.

    Returns
    -------
    tuple
        ``(merges, bytes_maps)`` where ``bytes_maps`` may be ``None``.

    Raises
    ------
    ValueError
        If the file format is invalid or the version is unsupported.
    """
    if not path.endswith(".tbm"):
        if Path(path).suffix != ".tbm":
            path += ".tbm"

    merges: list[tuple[int, int]] = []
    bytes_maps: list[int] | None = None
    remap_builder: list[int] = []

    with open(path, encoding="utf-8") as f:
        magic = f.readline().strip()

        # Parse version from header: "TinyBPE Model v<N>"
        if "v" in magic:
            version_str = magic.split("v")[-1]
            try:
                version = int(version_str)
            except ValueError as exc:
                raise ValueError(f"Invalid model file header: {magic}") from exc
        else:
            version = 0  # legacy

        if version > MODEL_VERSION:
            raise ValueError(f"Model file version {version} is newer than the supported version ({MODEL_VERSION})")

        if not magic.startswith("TinyBPE Model"):
            raise ValueError(f"Invalid model file header: {magic}")

        # Read byte remapping
        remap_line = f.readline().strip()
        if remap_line == "256":
            for _ in range(256):
                remap_builder.append(int(f.readline().strip()))
            bytes_maps = remap_builder
        elif remap_line == "0":
            bytes_maps = None
        else:
            raise ValueError(f"Unexpected remap count: {remap_line}")

        # Read merge pairs
        for line in f:
            line = line.strip()
            if not line:
                continue
            left, right = map(int, line.split())
            merges.append((left, right))

    return merges, bytes_maps


# ---------------------------------------------------------------------------
# .vocab — vocabulary file
# ---------------------------------------------------------------------------


def save_vocab(path: str, vocab: dict[int, bytes]) -> None:
    """Save vocabulary to a ``.vocab`` file.

    Format: tab-separated ``base64_encoded_bytes<TAB>rank``,
    compatible with the tiktoken .tiktoken file format.

    Parameters
    ----------
    path : str
        Output file path (``.vocab`` appended if missing).
    vocab : dict[int, bytes]
        Vocabulary mapping token IDs to byte sequences.
    """
    import base64

    if not path.endswith(".vocab"):
        path += ".vocab"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TinyBPE Vocabulary v{MODEL_VERSION}\n")
        for rank in sorted(vocab.keys()):
            encoded = base64.b64encode(vocab[rank]).decode("ascii")
            f.write(f"{encoded} {rank}\n")


def load_vocab(path: str) -> dict[int, bytes]:
    """Load a vocabulary file.

    Parameters
    ----------
    path : str
        Path to the ``.vocab`` file.

    Returns
    -------
    dict[int, bytes]
        Vocabulary mapping token IDs to byte sequences.
    """
    import base64

    vocab: dict[int, bytes] = {}

    with open(path, encoding="utf-8") as f:
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
            vocab[rank] = base64.b64decode(encoded)

    return vocab
