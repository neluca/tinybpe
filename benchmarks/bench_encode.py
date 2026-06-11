#!/usr/bin/env python3
"""Benchmark: BPE encoding speed."""

import time
from tinybpe import Trainer, Tokenizer


def main():
    # Train on moderate corpus
    text = "hello world the quick brown fox " * 2000
    print(f"Training on {len(text)} chars...")
    t0 = time.perf_counter()
    trainer = Trainer(text)
    n = trainer.train(500)
    train_time = time.perf_counter() - t0
    print(f"  {n} merges in {train_time:.3f}s")

    tok = Tokenizer(trainer.merges)

    # Benchmark encode
    test_texts = [
        "hello world " * 100,
        "the quick brown fox jumps over the lazy dog " * 50,
        "hello " * 1000,
    ]

    for i, t in enumerate(test_texts):
        t0 = time.perf_counter()
        for _ in range(100):
            tok.encode(t)
        elapsed = time.perf_counter() - t0
        print(f"  Test {i}: {len(t)} chars × 100 = {elapsed:.3f}s "
              f"({elapsed / 100 * 1000:.3f} ms/encode)")


if __name__ == "__main__":
    main()
