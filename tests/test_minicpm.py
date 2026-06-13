"""Tests for MiniCPM-converted .tbm model."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinybpe import Tokenizer, load_model

MODEL_PATH = "models/minicpm.tbm"

# Skip all tests if model file doesn't exist
pytestmark = pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason=f"{MODEL_PATH} not found. Run scripts/convert_minicpm.py first.",
)

SP = "▁"  # SentencePiece space marker


def preprocess(text: str) -> str:
    """Apply SentencePiece normalization: prepend SP, replace spaces."""
    if not text:
        return ""
    return SP + text.replace(" ", SP)


def postprocess(text: str) -> str:
    """Reverse SentencePiece normalization.

    MiniCPM's decoder does:
    1. Replace SP (U+2581) with space
    2. Strip exactly ONE leading space (not all leading spaces)
    """
    text = text.replace(SP, " ")
    if text.startswith(" "):
        text = text[1:]
    return text


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

ASCII_TEXTS = [
    "hello world",
    "Hello World! How are you?",
    "the quick brown fox jumps over the lazy dog",
    "Python programming is fun",
    "abcdefghijklmnopqrstuvwxyz",
    "a",
    "def hello(): return 'world'",
    "print(f'hello {name}')",
    "import numpy as np; x = [1, 2, 3]",
]

MULTILINGUAL_TEXTS = [
    "你好世界",
    "只是一个测试",
    "日本語テスト",
    "한국어 테스트",
    "Привет мир",
    "🌍🌎🌏",
]

CODE_TEXTS = [
    "def hello(): return 'world'",
    "class MyClass:\n    def __init__(self):\n        pass",
    "x = [1, 2, 3]; y = {a: 1, b: 2}",
]

EDGE_CASES = [
    "",
    " ",
    "   ",
    "a",
    "hello\nworld\ttest\rcr",
    "hello!@#$%^&*()world",
    "hello" * 100,
]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """Load the MiniCPM .tbm model once per test module."""
    merges, bytes_maps = load_model(MODEL_PATH)
    assert bytes_maps is None, "MiniCPM model should have no byte remapping"
    return Tokenizer(merges, bytes_maps=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMiniCPMModel:
    """Test the converted MiniCPM .tbm model for correctness."""

    def test_model_loads(self, tokenizer):
        """Model should load with reasonable vocab size."""
        assert tokenizer.n_vocab > 100000
        assert tokenizer.n_vocab < 200000

    def test_ascii_roundtrip(self, tokenizer):
        """ASCII texts should round-trip correctly."""
        for text in ASCII_TEXTS:
            pre = preprocess(text)
            ids = tokenizer.encode(pre)
            decoded = tokenizer.decode(ids)
            restored = postprocess(decoded)
            assert restored == text, f"Round-trip failed for {text!r}: got {restored!r}"

    def test_multilingual_roundtrip(self, tokenizer):
        """Multilingual texts should round-trip correctly."""
        for text in MULTILINGUAL_TEXTS:
            pre = preprocess(text)
            ids = tokenizer.encode(pre)
            decoded = tokenizer.decode(ids)
            restored = postprocess(decoded)
            assert restored == text, f"Round-trip failed for {text!r}: got {restored!r}"

    def test_code_roundtrip(self, tokenizer):
        """Code snippets should round-trip correctly."""
        for text in CODE_TEXTS:
            pre = preprocess(text)
            ids = tokenizer.encode(pre)
            decoded = tokenizer.decode(ids)
            restored = postprocess(decoded)
            assert restored == text, f"Round-trip failed for {text!r}: got {restored!r}"

    def test_edge_cases(self, tokenizer):
        """Edge cases should work correctly."""
        for text in EDGE_CASES:
            if text == "":
                ids = tokenizer.encode("")
                decoded = tokenizer.decode(ids)
                assert decoded == ""
            else:
                pre = preprocess(text)
                ids = tokenizer.encode(pre)
                decoded = tokenizer.decode(ids)
                restored = postprocess(decoded)
                assert restored == text, f"Round-trip failed for {text!r}: got {restored!r}"

    def test_empty_string(self, tokenizer):
        """Empty string should produce empty token list."""
        assert tokenizer.encode("") == []
        assert tokenizer.decode([]) == ""

    def test_single_char(self, tokenizer):
        """Single characters should round-trip."""
        for c in "abcXYZ012!@#":
            pre = preprocess(c)
            ids = tokenizer.encode(pre)
            decoded = tokenizer.decode(ids)
            restored = postprocess(decoded)
            assert restored == c, f"Failed for {c!r}: got {restored!r}"

    def test_special_chars(self, tokenizer):
        """Special characters like newlines should round-trip."""
        text = "hello\nworld\t\rtest"
        pre = preprocess(text)
        ids = tokenizer.encode(pre)
        decoded = tokenizer.decode(ids)
        restored = postprocess(decoded)
        assert restored == text

    def test_streaming_decode(self, tokenizer):
        """Streaming decode should match batch decode."""
        for text in ASCII_TEXTS[:5] + MULTILINGUAL_TEXTS[:3]:
            if not text:
                continue
            pre = preprocess(text)
            ids = tokenizer.encode(pre)
            parts: list[str] = []
            decoder = tokenizer.stream_decode(parts.append)
            for tid in ids:
                decoder(tid)
            full = "".join(parts)
            restored = postprocess(full)
            assert restored == text

    def test_long_text(self, tokenizer):
        """Long text should round-trip correctly."""
        text = "hello world " * 200
        pre = preprocess(text)
        ids = tokenizer.encode(pre)
        decoded = tokenizer.decode(ids)
        restored = postprocess(decoded)
        assert restored == text

    def test_vocab_size_consistency(self, tokenizer):
        """Vocab size should equal 256 + n_merges."""
        assert tokenizer.n_vocab == 256 + len(tokenizer.merges)

    def test_merge_dependency_order(self, tokenizer):
        """Each merge's operands should already exist."""
        merges = tokenizer.merges
        for idx, (left, right) in enumerate(merges):
            max_valid = 256 + idx
            assert left < max_valid, f"merge {idx}: left {left} >= {max_valid}"
            assert right < max_valid, f"merge {idx}: right {right} >= {max_valid}"

    def test_encoding_is_deterministic(self, tokenizer):
        """Same input should produce same tokens every time."""
        text = "hello world test 123"
        pre = preprocess(text)
        ids1 = tokenizer.encode(pre)
        ids2 = tokenizer.encode(pre)
        assert ids1 == ids2

    def test_spaces_roundtrip(self, tokenizer):
        """Multiple consecutive spaces should round-trip."""
        for n in range(1, 6):
            text = "a" + " " * n + "b"
            pre = preprocess(text)
            ids = tokenizer.encode(pre)
            decoded = tokenizer.decode(ids)
            restored = postprocess(decoded)
            assert restored == text, f"Failed for {n} spaces: {text!r} -> {restored!r}"
