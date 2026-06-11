"""Fuzz / property-based tests for TinyBPE tokenizer."""

import random

from tinybpe import Tokenizer, Trainer


def _random_unicode_string(length: int) -> str:
    """Generate a random string with various Unicode categories."""
    chars = []
    for _ in range(length):
        cat = random.randint(0, 7)
        if cat == 0:
            # ASCII
            chars.append(chr(random.randint(32, 126)))
        elif cat == 1:
            # 2-byte UTF-8 (Latin-1 Supplement)
            chars.append(chr(random.randint(0xA0, 0xFF)))
        elif cat == 2:
            # 3-byte UTF-8 (CJK Unified)
            chars.append(chr(random.randint(0x4E00, 0x9FFF)))
        elif cat == 3:
            # 4-byte UTF-8 (Emoji)
            chars.append(chr(random.randint(0x1F300, 0x1F6FF)))
        elif cat == 4:
            # Whitespace
            chars.append(random.choice([" ", "\t", "\n", "\r"]))
        elif cat == 5:
            # CJK punctuation
            chars.append(chr(random.randint(0x3000, 0x303F)))
        elif cat == 6:
            # Digits
            chars.append(chr(random.randint(0x30, 0x39)))
        else:
            # Random BMP
            chars.append(chr(random.randint(0x100, 0xFFF)))
    return "".join(chars)


class TestFuzz:
    """Property-based fuzz tests."""

    @classmethod
    def setup_class(cls):
        """Train a model on diverse text for fuzz testing."""
        training_text = (
            "hello world " * 500
            + "the quick brown fox jumps over the lazy dog " * 300
            + "测试中文汉字 " * 200
            + "emoji 🎉😀🔥 " * 100
            + "1234567890 " * 200
            + "ABCDEFGHIJKLMNOPQRSTUVWXYZ " * 100
            + "abcdefghijklmnopqrstuvwxyz " * 100
        )
        trainer = Trainer(training_text)
        trainer.train(200)
        cls.merges = trainer.merges

    def test_roundtrip_random(self):
        """Random strings should always round-trip through encode→decode."""
        tok = Tokenizer(self.merges)
        random.seed(42)

        for _ in range(100):
            text = _random_unicode_string(random.randint(1, 200))
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text, f"Failed on: {text!r}"

    def test_roundtrip_repeated(self):
        """Repeated random strings should round-trip."""
        tok = Tokenizer(self.merges)
        random.seed(123)

        for _ in range(20):
            base = _random_unicode_string(random.randint(1, 50))
            text = base * random.randint(1, 10)
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text

    def test_stream_decode_random(self):
        """Streaming decode should produce same result as batch decode."""
        tok = Tokenizer(self.merges)
        random.seed(99)

        for _ in range(50):
            text = _random_unicode_string(random.randint(1, 100))
            ids = tok.encode(text)

            # Batch decode
            batch = tok.decode(ids)

            # Streaming decode
            parts: list[str] = []
            decoder = tok.stream_decode(parts.append)
            for tid in ids:
                decoder(tid)
            streamed = "".join(parts)

            assert batch == streamed == text
