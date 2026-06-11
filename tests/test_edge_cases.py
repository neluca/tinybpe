"""Edge case tests for TinyBPE tokenizer and trainer."""

from tinybpe import Tokenizer, Trainer


class TestEdgeCases:
    """Boundary condition tests."""

    def test_empty_string(self):
        """Empty string encode/decode."""
        trainer = Trainer("hello world " * 100)
        trainer.train(20)
        tok = Tokenizer(trainer.merges)

        ids = tok.encode("")
        assert ids == []
        assert tok.decode(ids) == ""

    def test_single_byte(self):
        """Single byte input."""
        trainer = Trainer("hello world " * 100)
        trainer.train(20)
        tok = Tokenizer(trainer.merges)

        for c in "abcdefghijklmnopqrstuvwxyz":
            encoded = tok.encode(c)
            decoded = tok.decode(encoded)
            assert decoded == c

    def test_unicode_boundaries(self):
        """Test various Unicode boundary cases."""
        trainer = Trainer("hello world " * 100)
        trainer.train(30)
        tok = Tokenizer(trainer.merges)

        # 2-byte UTF-8 (Latin-1 Supplement)
        text = "éñü"  # éñü
        ids = tok.encode(text)
        assert tok.decode(ids) == text

        # 3-byte UTF-8 (CJK)
        text = "中文测试"
        ids = tok.encode(text)
        assert tok.decode(ids) == text

        # 4-byte UTF-8 (emoji)
        text = "😀🎉🔥"
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_long_repetition(self):
        """Long repeated text."""
        trainer = Trainer("hello world " * 100)
        trainer.train(20)
        tok = Tokenizer(trainer.merges)

        text = "hello " * 500
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_special_chars(self):
        """Newlines, tabs, and special characters."""
        trainer = Trainer("hello world " * 100)
        trainer.train(20)
        tok = Tokenizer(trainer.merges)

        text = "hello\nworld\t\rtest null"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_very_large_ids(self):
        """Decode with large token IDs (special tokens)."""
        trainer = Trainer("hello world " * 100)
        trainer.train(20)
        special = {"<eot>": 99999}
        tok = Tokenizer(trainer.merges, special_tokens=special)

        ids = tok.encode("<eot>")
        assert ids == [99999]
        decoded = tok.decode(ids)
        assert decoded == "<eot>"

    def test_mixed_special_and_normal(self):
        """Mix of special and normal tokens."""
        trainer = Trainer("hello world " * 100)
        trainer.train(20)
        special = {"<eot>": 1000, "<fim>": 1001}
        tok = Tokenizer(trainer.merges, special_tokens=special)

        text = "<eot> hello <fim> world"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_consecutive_specials(self):
        """Two special tokens in a row."""
        trainer = Trainer("hello world " * 100)
        trainer.train(20)
        special = {"<a>": 1000, "<b>": 1001}
        tok = Tokenizer(trainer.merges, special_tokens=special)

        text = "<a><b>"
        ids = tok.encode(text)
        assert ids == [1000, 1001]
        decoded = tok.decode(ids)
        assert decoded == text
