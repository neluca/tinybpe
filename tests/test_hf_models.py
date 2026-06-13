"""Tests for HuggingFace-converted .tbm models."""

from __future__ import annotations

import pytest

from tinybpe import Tokenizer, load_model

try:
    from huggingface_hub import hf_hub_download  # noqa: F401

    HAS_HF = True
except ImportError:
    HAS_HF = False

# Regex patterns for ByteLevel BPE pre-tokenization
PAT_GPT2 = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

HF_MODELS = [
    (
        "qwen35",
        "Qwen/Qwen3.5-0.8B",
        "Qwen 3.5",
        PAT_GPT2,
    ),
]

# ---------------------------------------------------------------------------
# Test texts
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

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_HF, reason="huggingface_hub not installed")
class TestHFModels:
    """Test HuggingFace-converted tokenizers for correctness."""

    @pytest.mark.parametrize(("model_key", "hf_id", "desc", "pattern"), HF_MODELS)
    def test_model_file_exists(self, model_key, hf_id, desc, pattern):
        """Model .tbm file should exist and be loadable."""
        path = f"tinybpe/models/{model_key}.tbm"
        merges, bm = load_model(path)
        assert len(merges) > 0
        # Byte remapping: either None (ID-remapped) or a list of 256
        if bm is not None:
            assert len(bm) == 256

    @pytest.mark.parametrize(("model_key", "hf_id", "desc", "pattern"), HF_MODELS)
    def test_ascii_roundtrip(self, model_key, hf_id, desc, pattern):
        """ASCII texts should round-trip correctly."""
        tok = Tokenizer.from_file(f"tinybpe/models/{model_key}.tbm", pat_str=pattern)
        for text in ASCII_TEXTS:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text, f"Round-trip failed for {text!r}"

    @pytest.mark.parametrize(("model_key", "hf_id", "desc", "pattern"), HF_MODELS)
    def test_multilingual_roundtrip(self, model_key, hf_id, desc, pattern):
        """Multilingual texts should round-trip correctly."""
        tok = Tokenizer.from_file(f"tinybpe/models/{model_key}.tbm", pat_str=pattern)
        for text in MULTILINGUAL_TEXTS:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text

    @pytest.mark.parametrize(("model_key", "hf_id", "desc", "pattern"), HF_MODELS)
    def test_vocab_size(self, model_key, hf_id, desc, pattern):
        """Vocab size should match the original tokenizer."""
        tok = Tokenizer.from_file(f"tinybpe/models/{model_key}.tbm", pat_str=pattern)
        assert tok.n_vocab == 256 + len(tok.merges)

    @pytest.mark.parametrize(("model_key", "hf_id", "desc", "pattern"), HF_MODELS)
    def test_streaming_decode(self, model_key, hf_id, desc, pattern):
        """Streaming decode should match batch decode."""
        tok = Tokenizer.from_file(f"tinybpe/models/{model_key}.tbm", pat_str=pattern)
        for text in ASCII_TEXTS[:5] + MULTILINGUAL_TEXTS[:3]:
            if not text:
                continue
            ids = tok.encode(text)
            parts: list[str] = []
            decoder = tok.stream_decode(parts.append)
            for tid in ids:
                decoder(tid)
            assert "".join(parts) == text

    @pytest.mark.parametrize(("model_key", "hf_id", "desc", "pattern"), HF_MODELS)
    def test_empty_string(self, model_key, hf_id, desc, pattern):
        """Empty string should produce empty token list."""
        tok = Tokenizer.from_file(f"tinybpe/models/{model_key}.tbm", pat_str=pattern)
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    @pytest.mark.parametrize(("model_key", "hf_id", "desc", "pattern"), HF_MODELS)
    def test_single_char(self, model_key, hf_id, desc, pattern):
        """Single characters should round-trip."""
        tok = Tokenizer.from_file(f"tinybpe/models/{model_key}.tbm", pat_str=pattern)
        for c in "abcXYZ012!@#":
            ids = tok.encode(c)
            decoded = tok.decode(ids)
            assert decoded == c

    @pytest.mark.parametrize(("model_key", "hf_id", "desc", "pattern"), HF_MODELS)
    def test_special_chars(self, model_key, hf_id, desc, pattern):
        """Special characters (newlines, tabs) should round-trip."""
        tok = Tokenizer.from_file(f"tinybpe/models/{model_key}.tbm", pat_str=pattern)
        text = "hello\nworld\t\rtest"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    @pytest.mark.parametrize(("model_key", "hf_id", "desc", "pattern"), HF_MODELS)
    def test_long_text(self, model_key, hf_id, desc, pattern):
        """Long text should round-trip correctly."""
        tok = Tokenizer.from_file(f"tinybpe/models/{model_key}.tbm", pat_str=pattern)
        text = "hello world " * 200
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text


@pytest.mark.skipif(not HAS_HF, reason="huggingface_hub not installed")
class TestQwen35Specific:
    """Qwen 3.5 specific tests."""

    def test_chinese_text(self):
        """Chinese text should encode and decode correctly."""
        tok = Tokenizer.from_file("tinybpe/models/qwen35.tbm", pat_str=PAT_GPT2)

        cn_texts = [
            "他是一个独自一人划着小船在墨西哥湾大海流打鱼的老人",
            "人工智能正在改变世界",
            "深度学习是机器学习的一个分支",
            "今天天气真好",
        ]
        for text in cn_texts:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text

    def test_code_text(self):
        """Code snippets should round-trip."""
        tok = Tokenizer.from_file("tinybpe/models/qwen35.tbm", pat_str=PAT_GPT2)

        code = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""
        ids = tok.encode(code)
        decoded = tok.decode(ids)
        assert decoded == code




@pytest.mark.skipif(not HAS_HF, reason="huggingface_hub not installed")
class TestDeepSeekSpecific:
    """DeepSeek-V4 tokenizer specific tests.

    DeepSeek-V4 uses standard ByteLevel BPE with ID-remapped token IDs.
    """

    def test_roundtrip_ascii(self):
        """ASCII text should round-trip."""
        tok = Tokenizer.from_file("tinybpe/models/deepseek-v4.tbm")
        for text in [
            "hello world",
            "Hello World! How are you?",
            "the quick brown fox jumps over the lazy dog",
            "Python programming is fun",
            "def hello(): return 'world'",
        ]:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text

    def test_roundtrip_chinese(self):
        """Chinese text should round-trip."""
        tok = Tokenizer.from_file("tinybpe/models/deepseek-v4.tbm")
        cn_texts = [
            "他是一个独自一人划着小船在墨西哥湾大海流打鱼的老人",
            "人工智能正在改变世界",
            "深度学习是机器学习的一个分支",
        ]
        for text in cn_texts:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text

    def test_roundtrip_code(self):
        """Code snippets should round-trip."""
        tok = Tokenizer.from_file("tinybpe/models/deepseek-v4.tbm")
        code = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        ids = tok.encode(code)
        decoded = tok.decode(ids)
        assert decoded == code

    def test_roundtrip_emoji(self):
        """Emoji should round-trip."""
        tok = Tokenizer.from_file("tinybpe/models/deepseek-v4.tbm")
        text = "👋😊🎉🔥💻"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_vocab_size(self):
        """Vocab size should be correct."""
        tok = Tokenizer.from_file("tinybpe/models/deepseek-v4.tbm")
        # DeepSeek-V4 has ~128K total tokens
        assert 127000 <= tok.n_vocab <= 129000
        assert tok.n_vocab == 256 + len(tok.merges)

    def test_bytes_maps_bijection(self):
        """Bytes are identity-mapped (no remap needed after ID remapping)."""
        from tinybpe import load_model

        _, bm = load_model("tinybpe/models/deepseek-v4.tbm")
        # ID-remapped models have bytes_maps=None
        assert bm is None

    def test_streaming_decode(self):
        """Streaming decode should match batch decode."""
        tok = Tokenizer.from_file("tinybpe/models/deepseek-v4.tbm")
        text = "hello world 你好 世界"
        ids = tok.encode(text)
        parts: list[str] = []
        decoder = tok.stream_decode(parts.append)
        for tid in ids:
            decoder(tid)
        assert "".join(parts) == text
