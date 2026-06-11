"""Unit tests for the C extension directly (bpe.Trainer, bpe.Tokenizer, bpe.BytesRemap)."""

import tinybpe.bpe as bpe


class TestCTrainer:
    """Tests for bpe.Trainer (C-level)."""

    def test_create(self):
        trainer = bpe.Trainer([b"hello", b"world"])
        assert trainer is not None

    def test_step(self):
        trainer = bpe.Trainer([b"hello", b"world"])
        result = trainer.step()
        assert result is not None
        pair, rank, freq = result
        assert isinstance(pair, tuple)
        assert rank >= 256
        assert freq >= 1

    def test_merges_accumulate(self):
        trainer = bpe.Trainer([b"hello world " * 50])
        for _ in range(10):
            result = trainer.step()
            if result is None:
                break
        assert trainer.n_merges > 0
        assert len(trainer.merges) == trainer.n_merges

    def test_load_merges(self):
        trainer1 = bpe.Trainer([b"hello world " * 50])
        trainer1.step()
        trainer1.step()
        merges = trainer1.merges

        trainer2 = bpe.Trainer([b"hello world " * 50])
        trainer2.load_merges(merges)
        # After loading, training continues from the loaded state
        result = trainer2.step()
        assert result is not None

    def test_empty_list_raises(self):
        try:
            bpe.Trainer([])
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for empty list")

    def test_non_bytes_raises(self):
        import pytest

        with pytest.raises(TypeError):
            bpe.Trainer(["not bytes"])  # type: ignore[list-item]


class TestCTokenizer:
    """Tests for bpe.Tokenizer (C-level)."""

    @classmethod
    def setup_class(cls):
        """Train a small model to get merges."""
        trainer = bpe.Trainer([b"hello world " * 200])
        for _ in range(50):
            r = trainer.step()
            if r is None:
                break
        cls.merges = trainer.merges

    def test_create(self):
        tok = bpe.Tokenizer(self.merges)
        assert tok is not None

    def test_encode_decode(self):
        tok = bpe.Tokenizer(self.merges)
        ids = tok.encode(b"hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert decoded == b"hello world"

    def test_encode_empty(self):
        tok = bpe.Tokenizer(self.merges)
        ids = tok.encode(b"")
        assert ids == []

    def test_decode_empty(self):
        tok = bpe.Tokenizer(self.merges)
        result = tok.decode([])
        assert result == b""

    def test_vocab_property(self):
        tok = bpe.Tokenizer(self.merges)
        vocab = tok.vocab
        assert isinstance(vocab, dict)
        assert len(vocab) == 256 + len(self.merges)
        # Base bytes 0-255
        assert vocab[65] == b"A"
        assert vocab[97] == b"a"

    def test_n_vocab(self):
        tok = bpe.Tokenizer(self.merges)
        assert tok.n_vocab == 256 + len(self.merges)

    def test_special_tokens(self):
        special = {b"<eot>": 1000, b"<fim>": 1001}
        tok = bpe.Tokenizer(self.merges, special)
        # Special token match
        ids = tok.encode(b"<eot>")
        assert ids == [1000]
        # Decode includes special tokens
        decoded = tok.decode([1000, 5])
        assert b"<eot>" in decoded

    def test_cache_decode(self):
        tok = bpe.Tokenizer(self.merges)
        ids = tok.encode(b"hello")
        tok.cache_clean()
        result_parts = []
        for tid in ids:
            part = tok.cache_decode(tid)
            if part is not None:
                result_parts.append(part)
        assert b"".join(result_parts) == b"hello"

    def test_cache_clean(self):
        tok = bpe.Tokenizer(self.merges)
        tok.cache_clean()  # Should not raise


class TestCBytesRemap:
    """Tests for bpe.BytesRemap (C-level)."""

    def test_identity(self):
        remap = bpe.BytesRemap(list(range(256)))
        result = remap(b"hello")
        assert result == b"hello"

    def test_reverse(self):
        reverse_map = list(reversed(range(256)))
        remap = bpe.BytesRemap(reverse_map)
        result = remap(b"\x00\x01\x02")
        assert result == b"\xff\xfe\xfd"

    def test_invalid_size(self):
        try:
            bpe.BytesRemap([0] * 255)  # Must be 256
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")

    def test_out_of_range(self):
        try:
            bpe.BytesRemap([256] + [0] * 255)  # 256 is out of range
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")
