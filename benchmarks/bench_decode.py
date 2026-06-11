#!/usr/bin/env python3
"""Benchmark: BPE decoding speed."""

import time
from tinybpe import Trainer, Tokenizer


def main():
    text = "hello world the quick brown fox " * 2000
    trainer = Trainer(text)
    n = trainer.train(500)
    print(f"Trained {n} merges")

    tok = Tokenizer(trainer.merges)

    test_texts = [
        "hello world " * 100,
        "the quick brown fox jumps over the lazy dog " * 50,
        "hello " * 1000,
    ]

    for i, t in enumerate(test_texts):
        ids = tok.encode(t)

        t0 = time.perf_counter()
        for _ in range(1000):
            tok.decode(ids)
        elapsed = time.perf_counter() - t0
        print(f"  Test {i}: {len(ids)} ids × 1000 = {elapsed:.3f}s "
              f"({elapsed / 1000 * 1e6:.1f} µs/decode)")


if __name__ == "__main__":
    main()
