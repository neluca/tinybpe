#!/usr/bin/env python3
"""Streaming decode example."""

from tinybpe import Trainer, Tokenizer


def main():
    # Train a small model
    trainer = Trainer("hello world " * 500)
    trainer.train(50)
    tok = Tokenizer(trainer.merges)

    text = "hello world hello hello"
    ids = tok.encode(text)

    # Streaming decode: process one token at a time
    print("Streaming decode:")
    parts = []

    def on_fragment(s: str) -> None:
        print(f"  fragment: {s!r}")
        parts.append(s)

    decoder = tok.stream_decode(on_fragment)
    for tid in ids:
        decoder(tid)

    result = "".join(parts)
    print(f"\nResult: {result!r}")
    print(f"Match: {result == text}")


if __name__ == "__main__":
    main()
