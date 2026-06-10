"""Benchmark: BPE encoding throughput across input sizes."""

import time
import sys
import os

# Add parent directory to path for local runs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tinybpe import bpe


def benchmark_encode(merges, text_bytes: bytes, iterations: int = 10) -> float:
    """Measure encode time per call in milliseconds."""
    tokenizer = bpe.Tokenizer(merges)

    # Warm up
    tokenizer.encode(text_bytes)

    start = time.perf_counter()
    for _ in range(iterations):
        tokenizer.encode(text_bytes)
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000  # ms per call


def main():
    # A realistic set of merges
    merges = [
        (104, 101), (256, 108), (257, 108), (258, 111), (259, 32),
        (119, 111), (261, 114), (262, 108), (263, 100),
        (116, 104), (264, 101), (265, 32), (266, 105), (267, 115),
    ]

    sizes = [1024, 10240, 102400, 1048576]
    print("Encode Benchmark")
    print("=" * 60)
    print(f"{'Input Size':>12} | {'Time (ms)':>10} | {'Throughput':>15}")
    print("-" * 60)

    for size in sizes:
        text = b"hello world this is a benchmark test. " * (size // 45)
        text = text[:size]
        ms = benchmark_encode(merges, text)
        mb_per_sec = (size / 1024 / 1024) / (ms / 1000)
        print(f"{size:>12,} | {ms:>10.2f} | {mb_per_sec:>12.2f} MB/s")


if __name__ == "__main__":
    main()
