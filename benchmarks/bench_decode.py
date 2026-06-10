"""Benchmark: BPE decoding throughput across token list sizes."""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tinybpe import bpe


def benchmark_decode(merges, ids: list[int], iterations: int = 10) -> float:
    """Measure decode time per call in milliseconds."""
    tokenizer = bpe.Tokenizer(merges)

    # Warm up
    tokenizer.decode(ids)

    start = time.perf_counter()
    for _ in range(iterations):
        tokenizer.decode(ids)
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000  # ms per call


def main():
    merges = [
        (104, 101), (256, 108), (257, 108), (258, 111), (259, 32),
        (119, 111), (261, 114), (262, 108), (263, 100),
    ]

    tokenizer = bpe.Tokenizer(merges)

    sizes = [100, 1000, 10000, 100000]
    print("Decode Benchmark")
    print("=" * 60)
    print(f"{'# Tokens':>12} | {'Time (ms)':>10} | {'Tokens/ms':>12}")
    print("-" * 60)

    for size in sizes:
        # Generate text and encode to get realistic token sequence
        text = b"hello world " * (max(size // 4, 1))
        text = text[:size * 2]  # generous buffer
        ids = tokenizer.encode(text)

        ms = benchmark_decode(merges, ids)
        tokens_per_ms = len(ids) / ms
        print(f"{len(ids):>12,} | {ms:>10.3f} | {tokens_per_ms:>12.1f}")


if __name__ == "__main__":
    main()
