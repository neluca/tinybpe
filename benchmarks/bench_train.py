"""Benchmark: BPE training throughput."""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tinybpe import bpe


def benchmark_train(text_bytes: bytes, n_steps: int) -> tuple[float, int]:
    """Train for n_steps, returning time per step in ms and merges produced."""
    trainer = bpe.Trainer([text_bytes])

    start = time.perf_counter()
    merges = 0
    for _ in range(n_steps):
        result = trainer.step()
        if result is None:
            break
        merges += 1
    elapsed = time.perf_counter() - start

    if merges == 0:
        return 0.0, 0

    return (elapsed / merges) * 1000, merges  # ms per step


def main():
    print("Training Benchmark")
    print("=" * 60)
    print(f"{'Input Size':>12} | {'Steps':>8} | {'ms/step':>10} | {'Total (s)':>10}")
    print("-" * 60)

    sizes = [10240, 102400, 1048576]
    for size in sizes:
        text = b"hello world " * (size // 12)
        text = text[:size]
        n_steps = min(200, size // 2)

        ms_per_step, actual_steps = benchmark_train(text, n_steps)
        total_s = (ms_per_step * actual_steps) / 1000

        print(f"{size:>12,} | {actual_steps:>8} | {ms_per_step:>10.2f} | {total_s:>10.2f}")


if __name__ == "__main__":
    main()
