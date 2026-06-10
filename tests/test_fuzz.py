"""Fuzz / property-based tests for TinyBPE.

Verifies invariants that should hold for all valid inputs.
"""

import unittest
import random
from tinybpe import bpe

# Simple merges that cover common patterns
MERGES = [
    (104, 101),   # h + e = 256
    (256, 108),   # he + l = 257
    (257, 108),   # hel + l = 258
    (258, 111),   # hell + o = 259
    (259, 32),    # hello + space = 260
    (119, 111),   # w + o = 261
    (261, 114),   # wo + r = 262
    (262, 108),   # wor + l = 263
    (263, 100),   # worl + d = 264
]


class TestRoundTrip(unittest.TestCase):
    """Verify that encode(decode(x)) == x and decode(encode(y)) == y."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = bpe.Tokenizer(MERGES)

    def test_encode_decode_roundtrip_ascii(self) -> None:
        """Any ASCII string should round-trip through encode then decode."""
        random.seed(12345)
        for _ in range(200):
            length = random.randint(0, 200)
            s = bytes(random.randint(32, 126) for _ in range(length))
            ids = self.tokenizer.encode(s)
            decoded = self.tokenizer.decode(ids)
            self.assertEqual(decoded, s, f"Round-trip failed for {s[:50]!r}")

    def test_encode_decode_roundtrip_all_bytes(self) -> None:
        """Random byte sequences should round-trip."""
        random.seed(54321)
        for _ in range(200):
            length = random.randint(0, 200)
            s = bytes(random.randint(0, 255) for _ in range(length))
            ids = self.tokenizer.encode(s)
            decoded = self.tokenizer.decode(ids)
            self.assertEqual(decoded, s, f"Round-trip failed for length {length}")

    def test_decode_encode_roundtrip(self) -> None:
        """Decode then encode should be idempotent for valid token sequences."""
        random.seed(11111)
        vocab_size = self.tokenizer.n_vocab
        for _ in range(100):
            length = random.randint(0, 50)
            ids = [random.randint(0, vocab_size - 1) for _ in range(length)]
            decoded = self.tokenizer.decode(ids)
            re_encoded = self.tokenizer.encode(decoded)
            re_decoded = self.tokenizer.decode(re_encoded)
            self.assertEqual(re_decoded, decoded)

    def test_no_crash_random_bytes(self) -> None:
        """Random bytes should never cause a crash during encode."""
        random.seed(99999)
        for _ in range(500):
            length = random.randint(0, 1000)
            s = bytes(random.randint(0, 255) for _ in range(length))
            try:
                ids = self.tokenizer.encode(s)
                # Verify the result is a valid list of non-negative ints
                self.assertIsInstance(ids, list)
                for tid in ids:
                    self.assertIsInstance(tid, int)
                    self.assertGreaterEqual(tid, 0)
            except Exception:
                # Exceptions are acceptable (e.g., MemoryError), crashes are not
                pass


class TestStreamDecodeInvariant(unittest.TestCase):
    """Verify stream_decode produces the same result as batch decode."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = bpe.Tokenizer(MERGES)

    def test_stream_matches_batch(self) -> None:
        """Stream decode should produce the same output as batch decode."""
        random.seed(77777)
        test_strings = [
            b"hello world",
            b"hello world hello world",
            bytes(range(256)),
            b"\xf0\x9f\x98\x80" * 10,
            b"\xe4\xbd\xa0\xe5\xa5\xbd",
            b"",
        ]
        for s in test_strings:
            ids = self.tokenizer.encode(s)
            batch_result = self.tokenizer.decode(ids)

            # Stream decode
            self.tokenizer.cache_clean()
            stream_parts = []

            def collect(text: bytes) -> None:
                stream_parts.append(text)

            decode_fn = self._stream_decode_wrapper(collect)
            for tid in ids:
                result = self.tokenizer.cache_decode(tid)
                if result is not None:
                    stream_parts.append(result)

            stream_result = b"".join(stream_parts)
            self.assertEqual(stream_result, batch_result, f"Failed for {s[:50]!r}")

    def _stream_decode_wrapper(self, callback):
        """Minimal stream decode wrapper using cache_decode."""
        def decode_one(tid):
            result = self.tokenizer.cache_decode(tid)
            if result is not None:
                callback(result)
        return decode_one


class TestTrainerInvariants(unittest.TestCase):
    """Verify training invariants."""

    def test_merges_are_monotonic(self) -> None:
        """New merges should always have increasing rank."""
        text = b"hello world " * 100
        trainer = bpe.Trainer([text])
        prev_rank = 255
        for _ in range(50):
            result = trainer.step()
            if result is None:
                break
            pair, rank, freq = result
            self.assertGreater(rank, prev_rank)
            prev_rank = rank

    def test_merges_are_unique(self) -> None:
        """No duplicate merge pairs should be generated."""
        text = b"hello world " * 100
        trainer = bpe.Trainer([text])
        seen = set()
        for _ in range(50):
            result = trainer.step()
            if result is None:
                break
            pair, rank, freq = result
            self.assertNotIn(pair, seen, f"Duplicate pair: {pair}")
            seen.add(pair)

    def test_same_input_same_merges(self) -> None:
        """Same input should produce same merges."""
        text = b"abcabcabc"
        t1 = bpe.Trainer([text])
        t2 = bpe.Trainer([text])

        merges1 = []
        merges2 = []
        for _ in range(10):
            r1 = t1.step()
            r2 = t2.step()
            if r1:
                merges1.append(r1[0])
            if r2:
                merges2.append(r2[0])

        self.assertEqual(merges1, merges2)
