def bpe_save_merges(file_prefix: str, merges: list[tuple[int, int]]):
    bpe_file = file_prefix + ".tinybpe"

    with open(bpe_file, 'w') as f:
        f.write("tinybpe model\n")
        f.write(f"{len(merges)}\n")

        for p1, p2 in merges:
            f.write(f"{p1} {p2}\n")


def bpe_save_remaps(file_prefix: str, remaps: list[int]):
    remaps_file = file_prefix + ".map"
    with open(remaps_file, 'w') as f:
        f.write("tinybpe remaps\n")
        for i in remaps:
            f.write(f"{i}\n")


def bpe_load_remaps(remaps_file: str) -> list[int]:
    assert remaps_file.endswith(".map")
    remaps = []
    with open(remaps_file, 'r', encoding="utf-8") as f:
        magic = f.readline().strip()
        assert magic == "tinybpe remaps"
        for line in f:
            remaps.append(int(line))
    return remaps
