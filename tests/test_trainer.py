"""Tests for the TinyBPE Trainer."""

import regex as re
from tinybpe import Trainer, Tokenizer, save_model, load_model
from pathlib import Path

TESTS_DIR = Path(__file__).parent


class TestTrainer:
    """Tests for the Trainer class."""

    def test_basic_training(self):
        text = "hello world hello world hello"
        trainer = Trainer(text)
        result = trainer.step()
        assert result is not None
        pair, rank, freq = result
        # 'he' or 'll' or 'lo' etc. — any valid merge
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert rank >= 256
        assert freq >= 1

    def test_train_n_merges(self):
        text = ("hello world " * 500)
        trainer = Trainer(text)
        n = trainer.train(20)  # "hello world " * 500 can produce ~24 merges
        assert n == 20
        assert trainer.n_merges == 20

    def test_train_exhaustion(self):
        """Training should stop early if no more pairs exist."""
        text = "ab"  # Only one pair
        trainer = Trainer(text)
        n = trainer.train(1000)  # Ask for more than possible
        assert n <= 1  # Should stop after at most 1 merge

    def test_merges_property(self):
        text = "hello world " * 100
        trainer = Trainer(text)
        trainer.train(10)
        merges = trainer.merges
        assert len(merges) == 10
        for pair in merges:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_save_load(self):
        text = "hello world " * 100
        trainer = Trainer(text)
        trainer.train(30)
        save_path = str(TESTS_DIR / "t_trainer")
        trainer.save(save_path)
        merges, bm = load_model(save_path + ".tbm")
        assert trainer.merges == merges
        assert bm is None

    def test_callback(self):
        steps: list = []

        def on_step(step, total, pair, rank, freq):
            steps.append((step, pair, rank, freq))

        text = "hello world " * 100
        trainer = Trainer(text, callback=on_step)
        trainer.train(10)
        assert len(steps) == 10
        assert steps[0][0] == 1  # step count
        assert steps[-1][0] == 10

    def test_preprocess(self):
        text = "hello world   testing"

        def preprocess(t: str) -> list[bytes]:
            # Split on whitespace
            return [chunk.encode("utf-8") for chunk in t.split()]

        trainer = Trainer(text, preprocess=preprocess)
        result = trainer.step()
        assert result is not None

    def test_load_merges_for_continue(self):
        text = "hello world " * 500
        trainer1 = Trainer(text)
        trainer1.train(20)

        # Continue training from a fresh copy of the same data
        trainer2 = Trainer(text)
        trainer2.load_merges(trainer1.merges)
        # After loading, the state should have the merges applied
        assert trainer2.n_merges == 20
        # We can try to train more, but the data may be exhausted
        n = trainer2.train(5)
        assert trainer2.n_merges == 20 + n

    def test_tokenizer_from_trainer(self):
        text = "hello world " * 100
        trainer = Trainer(text)
        trainer.train(30)

        tok = Tokenizer(trainer.merges)
        original = "hello world hi"
        ids = tok.encode(original)
        decoded = tok.decode(ids)
        assert decoded == original


class TestTrainerEdgeCases:
    """Edge case tests for the Trainer."""

    def test_empty_text(self):
        """Empty text should produce a valid but idle trainer."""
        try:
            trainer = Trainer("")
        except ValueError:
            # Expected: empty list not allowed
            pass

    def test_single_char(self):
        """Single character training should find merges."""
        text = "a" * 100
        trainer = Trainer(text)
        n = trainer.train(10)
        # Repeated "a" can produce many merges: aa→256, 256256→257, etc.
        assert n >= 1

    def test_two_chars(self):
        """Training on alternating characters."""
        text = "ab" * 500
        trainer = Trainer(text)
        n = trainer.train(10)
        assert n >= 1  # Should find at least one merge

    def test_non_utf8_preprocess(self):
        """Preprocess that returns raw bytes."""
        text = "hello world"
        trainer = Trainer(text, preprocess=lambda t: [b"hello", b"world"])
        result = trainer.step()
        assert result is not None
