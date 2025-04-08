def _save_bpe_merges(file_prefix: str, merges: list[tuple[int, int]]):
    bpe_file = file_prefix + ".tinybpe"

    with open(bpe_file, 'w') as f:
        f.write("tinybpe model\n")
        f.write(f"{len(merges)}\n")

        for p1, p2 in merges:
            f.write(f"{p1} {p2}\n")


def _save_bpe_vocab(file_prefix: str, vocab: dict[int, bytes]):
    vocab_file = file_prefix + ".vocab"

    with open(vocab_file, 'w') as f:
        f.write("tinybpe vocab\n")
        f.write(f"{len(vocab)}\n")

        for rank, text_bytes in vocab.items():
            f.write(f"{rank}: {text_bytes}\n")


def load_bpe_file(bpe_file: str) -> list[tuple[int, int]]:
    assert bpe_file.endswith(".tinybpe")
    merges = []

    with open(bpe_file, 'r', encoding="utf-8") as f:
        magic = f.readline().strip()
        assert magic == "tinybpe model"
        f.readline()
        for line in f:
            p1, p2 = map(int, line.split())
            merges.append((p1, p2))

    return merges
