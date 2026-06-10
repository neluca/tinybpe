"""Edge case and boundary tests for TinyBPE tokenizer and trainer."""

import unittest
from tinybpe import bpe


class TestEdgeCases(unittest.TestCase):
    """Tests for boundary conditions and edge cases."""

    def setUp(self) -> None:
        self.merges = [(104, 101), (256, 108), (257, 108),
                        (258, 111), (259, 32), (119, 111),
                        (260, 114), (261, 108), (262, 100)]

    def test_empty_encode(self) -> None:
        """Encoding empty bytes should return an empty list."""
        tokenizer = bpe.Tokenizer(self.merges)
        ids = tokenizer.encode(b"")
        self.assertEqual(ids, [])

    def test_empty_decode(self) -> None:
        """Decoding an empty list should return empty bytes."""
        tokenizer = bpe.Tokenizer(self.merges)
        result = tokenizer.decode([])
        self.assertEqual(result, b"")

    def test_single_byte(self) -> None:
        """Encoding and decoding a single byte should round-trip."""
        tokenizer = bpe.Tokenizer(self.merges)
        for byte_val in [b"a", b"\xff", b"\x00", b"\n", b"\t"]:
            ids = tokenizer.encode(byte_val)
            decoded = tokenizer.decode(ids)
            self.assertEqual(decoded, byte_val, f"Failed for {byte_val!r}")

    def test_all_byte_values_roundtrip(self) -> None:
        """All 256 byte values should round-trip through encode/decode."""
        tokenizer = bpe.Tokenizer(self.merges)
        all_bytes = bytes(range(256))
        ids = tokenizer.encode(all_bytes)
        decoded = tokenizer.decode(ids)
        self.assertEqual(decoded, all_bytes)

    def test_max_valid_id_decode(self) -> None:
        """Decoding the maximum valid token ID should succeed."""
        tokenizer = bpe.Tokenizer(self.merges)
        vocab_size = tokenizer.n_vocab
        # All IDs from 0 to vocab_size-1 should decode without error
        for tid in [0, 255, vocab_size - 1]:
            result = tokenizer.decode([tid])
            self.assertIsInstance(result, bytes)

    def test_single_id_roundtrip(self) -> None:
        """Each individual token ID should decode to its bytes."""
        tokenizer = bpe.Tokenizer(self.merges)
        vocab = tokenizer.vocab
        for tid in sorted(vocab.keys()):
            result = tokenizer.decode([tid])
            expected = vocab[tid]
            self.assertEqual(result, expected, f"Mismatch for ID {tid}")

    def test_multibyte_utf8_roundtrip(self) -> None:
        """Multi-byte UTF-8 sequences should encode/decode correctly."""
        tokenizer = bpe.Tokenizer(self.merges)
        test_strings = [
            b"\xc2\xa9",           # U+00A9 (2 bytes)
            b"\xe2\x82\xac",       # U+20AC (3 bytes)
            b"\xf0\x9f\x98\x80",   # U+1F600 (4 bytes)
            b"Hello, world!",
            b"\xe4\xbd\xa0\xe5\xa5\xbd",  # 你好
        ]
        for s in test_strings:
            ids = tokenizer.encode(s)
            decoded = tokenizer.decode(ids)
            self.assertEqual(decoded, s, f"Failed for {s!r}")

    def test_repeated_pattern(self) -> None:
        """A repeated pattern should encode consistently."""
        tokenizer = bpe.Tokenizer(self.merges)
        pattern = b"hello world"
        single_ids = tokenizer.encode(pattern)
        triple = pattern * 3
        triple_ids = tokenizer.encode(triple)
        self.assertEqual(triple_ids, single_ids * 3)

    def test_trainer_empty_bytes_list(self) -> None:
        """Trainer should raise on empty list."""
        with self.assertRaises(Exception):
            bpe.Trainer([])

    def test_tokenizer_empty_merges(self) -> None:
        """Tokenizer should raise on empty merges list."""
        with self.assertRaises(Exception):
            bpe.Tokenizer([])

    def test_trainer_step_until_done(self) -> None:
        """Trainer step should eventually return None when no more pairs."""
        trainer = bpe.Trainer([b"abc", b"def"])
        steps = 0
        while True:
            result = trainer.step()
            if result is None:
                break
            steps += 1
            if steps > 100:  # Safety limit
                break
        # Should have made some progress then stopped
        self.assertGreater(steps, 0)

    def test_continue_training(self) -> None:
        """Loading merges and continuing training should work."""
        trainer = bpe.Trainer([b"hello world"])
        trainer.load_merges([(104, 101)])  # h+e = he
        self.assertEqual(trainer.n_merges, 1)
        result = trainer.step()
        if result is not None:  # There may be more merges
            pair, rank, freq = result
            self.assertIsInstance(pair, tuple)
            self.assertIsInstance(rank, int)
            self.assertIsInstance(freq, int)


class TestBytesRemap(unittest.TestCase):
    """Tests for BytesRemap functionality."""

    def test_identity_remap(self) -> None:
        """Identity remap should not change bytes."""
        r = list(range(256))
        mapper = bpe.BytesRemap(r)
        self.assertEqual(mapper(b"hello"), b"hello")
        self.assertEqual(mapper(b""), b"")
        self.assertEqual(mapper(bytes(range(256))), bytes(range(256)))

    def test_reverse_remap(self) -> None:
        """Reverse remap should invert byte values."""
        r = list(reversed(range(256)))
        mapper = bpe.BytesRemap(r)
        s = b"abcdef"
        result = mapper(s)
        expected = bytes(255 - b for b in s)
        self.assertEqual(result, expected)

    def test_roundtrip(self) -> None:
        """Remap + inverse remap should produce identity."""
        r = list(range(256))
        r_inv = list(range(256))
        # Scramble the mapping
        import random
        random.seed(42)
        random.shuffle(r)
        for i, v in enumerate(r):
            r_inv[v] = i

        mapper = bpe.BytesRemap(r)
        inv_mapper = bpe.BytesRemap(r_inv)
        original = bytes(range(256))
        self.assertEqual(inv_mapper(mapper(original)), original)


class TestModelIOEdgeCases(unittest.TestCase):
    """Tests for model file I/O edge cases."""

    def test_load_invalid_extension(self) -> None:
        """Load should reject non-.tinymodel files."""
        from tinybpe._model_io import load_bpe_model
        with self.assertRaises(ValueError):
            load_bpe_model("test.txt")

    def test_bpe_param_roundtrip(self) -> None:
        """BPEParam should carry data correctly."""
        from tinybpe._model_io import BPEParam
        p = BPEParam(bytes_maps=list(range(256)), merges=[(1, 2), (3, 4)])
        self.assertEqual(len(p.bytes_maps), 256)
        self.assertEqual(len(p.merges), 2)
        self.assertEqual(p.merges[0], (1, 2))
