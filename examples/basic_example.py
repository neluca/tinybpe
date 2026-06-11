#!/usr/bin/env python3
"""Basic TinyBPE usage example."""

from tinybpe import Trainer, Tokenizer


def main():
    # 1. Train a BPE model on some text
    print("Training...")
    trainer = Trainer("hello world " * 500)
    n = trainer.train(100)
    print(f"  Learned {n} merges")

    # 2. Save the model
    trainer.save("example_model")
    print("  Saved example_model.tbm")

    # 3. Create a tokenizer from the trained model
    tok = Tokenizer(trainer.merges)

    # 4. Encode text
    text = "hello world hello"
    ids = tok.encode(text)
    print(f"\nEncode: {text!r}")
    print(f"  → {ids}")

    # 5. Decode back
    decoded = tok.decode(ids)
    print(f"Decode: {ids}")
    print(f"  → {decoded!r}")

    # 6. Verify round-trip
    assert text == decoded, "Round-trip failed!"
    print("\n✓ Round-trip OK")

    # 7. Show some vocabulary
    print(f"\nVocabulary size: {tok.n_vocab}")
    for tid, token_bytes in sorted(tok.vocab.items())[256:260]:
        print(f"  {tid}: {token_bytes!r}")


if __name__ == "__main__":
    main()
