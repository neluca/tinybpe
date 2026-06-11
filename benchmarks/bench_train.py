#!/usr/bin/env python3
"""Benchmark: BPE training speed."""

import time
from tinybpe import Trainer


def main():
    sizes = [1000, 5000, 20000]

    for size in sizes:
        text = "hello world the quick brown fox jumps over the lazy dog " * size
        print(f"\nTraining on {len(text):,} chars...")

        t0 = time.perf_counter()
        trainer = Trainer(text)
        n = trainer.train(500)
        elapsed = time.perf_counter() - t0

        print(f"  {n} merges in {elapsed:.3f}s "
              f"({len(text) / elapsed:.0f} chars/s)")


if __name__ == "__main__":
    main()
