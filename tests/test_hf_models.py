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
        "qwen25",
        "Qwen/Qwen2.5-0.5B",
        "Qwen 2.5",
        PAT_GPT2,
    ),
    (
        "phi2",
        "microsoft/phi-2",
        "Phi-2",
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

    @pytest.mark.parametrize("model_key,hf_id,desc,pattern", HF_MODELS)
    def test_model_file_exists(self, model_key, hf_id, desc, pattern):
        """Model .tbm file should exist and be loadable."""
        path = f"models/{model_key}.tbm"
        merges, bm = load_model(path)
        assert len(merges) > 0
        # ByteLevel tokenizers always have byte remapping
        assert bm is not None
        assert len(bm) == 256

    @pytest.mark.parametrize("model_key,hf_id,desc,pattern", HF_MODELS)
    def test_ascii_roundtrip(self, model_key, hf_id, desc, pattern):
        """ASCII texts should round-trip correctly."""
        tok = Tokenizer.from_file(f"models/{model_key}.tbm", pat_str=pattern)
        for text in ASCII_TEXTS:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text, f"Round-trip failed for {text!r}"

    @pytest.mark.parametrize("model_key,hf_id,desc,pattern", HF_MODELS)
    def test_multilingual_roundtrip(self, model_key, hf_id, desc, pattern):
        """Multilingual texts should round-trip correctly."""
        tok = Tokenizer.from_file(f"models/{model_key}.tbm", pat_str=pattern)
        for text in MULTILINGUAL_TEXTS:
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text

    @pytest.mark.parametrize("model_key,hf_id,desc,pattern", HF_MODELS)
    def test_vocab_size(self, model_key, hf_id, desc, pattern):
        """Vocab size should match the original tokenizer."""
        tok = Tokenizer.from_file(f"models/{model_key}.tbm", pat_str=pattern)
        assert tok.n_vocab == 256 + len(tok.merges)

    @pytest.mark.parametrize("model_key,hf_id,desc,pattern", HF_MODELS)
    def test_streaming_decode(self, model_key, hf_id, desc, pattern):
        """Streaming decode should match batch decode."""
        tok = Tokenizer.from_file(f"models/{model_key}.tbm", pat_str=pattern)
        for text in ASCII_TEXTS[:5] + MULTILINGUAL_TEXTS[:3]:
            if not text:
                continue
            ids = tok.encode(text)
            parts: list[str] = []
            decoder = tok.stream_decode(parts.append)
            for tid in ids:
                decoder(tid)
            assert "".join(parts) == text

    @pytest.mark.parametrize("model_key,hf_id,desc,pattern", HF_MODELS)
    def test_empty_string(self, model_key, hf_id, desc, pattern):
        """Empty string should produce empty token list."""
        tok = Tokenizer.from_file(f"models/{model_key}.tbm", pat_str=pattern)
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    @pytest.mark.parametrize("model_key,hf_id,desc,pattern", HF_MODELS)
    def test_single_char(self, model_key, hf_id, desc, pattern):
        """Single characters should round-trip."""
        tok = Tokenizer.from_file(f"models/{model_key}.tbm", pat_str=pattern)
        for c in "abcXYZ012!@#":
            ids = tok.encode(c)
            decoded = tok.decode(ids)
            assert decoded == c

    @pytest.mark.parametrize("model_key,hf_id,desc,pattern", HF_MODELS)
    def test_special_chars(self, model_key, hf_id, desc, pattern):
        """Special characters (newlines, tabs) should round-trip."""
        tok = Tokenizer.from_file(f"models/{model_key}.tbm", pat_str=pattern)
        text = "hello\nworld\t\rtest"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    @pytest.mark.parametrize("model_key,hf_id,desc,pattern", HF_MODELS)
    def test_long_text(self, model_key, hf_id, desc, pattern):
        """Long text should round-trip correctly."""
        tok = Tokenizer.from_file(f"models/{model_key}.tbm", pat_str=pattern)
        text = "hello world " * 200
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text


@pytest.mark.skipif(not HAS_HF, reason="huggingface_hub not installed")
class TestQwen25Specific:
    """Qwen 2.5 specific tests."""

    def test_chinese_text(self):
        """Chinese text should encode and decode correctly."""
        tok = Tokenizer.from_file("models/qwen25.tbm", pat_str=PAT_GPT2)

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
        tok = Tokenizer.from_file("models/qwen25.tbm", pat_str=PAT_GPT2)

        code = '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)'''
        ids = tok.encode(code)
        decoded = tok.decode(ids)
        assert decoded == code


@pytest.mark.skipif(not HAS_HF, reason="huggingface_hub not installed")
class TestPhi2Specific:
    """Phi-2 specific tests."""

    def test_code_roundtrip(self):
        """Code snippets should round-trip with Phi-2."""
        tok = Tokenizer.from_file("models/phi2.tbm", pat_str=PAT_GPT2)

        code = "def hello(): return 'world'"
        ids = tok.encode(code)
        decoded = tok.decode(ids)
        assert decoded == code

    def test_english_text(self):
        """English text should round-trip accurately."""
        tok = Tokenizer.from_file("models/phi2.tbm", pat_str=PAT_GPT2)

        text = "The quick brown fox jumps over the lazy dog."
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text
        # Phi-2 is GPT-2 based, should produce ~9 tokens for this sentence
        assert 7 <= len(ids) <= 12


@pytest.mark.skipif(not HAS_HF, reason="huggingface_hub not installed")
class TestDeepSeekSpecific:
    """DeepSeek tokenizer specific tests.

    DeepSeek uses a non-standard byte-to-unicode mapping that omits
    13 Latin-1 supplement characters (accents).  These correspond to
    byte values 0xC0-0xC1 and 0xF5-0xFF, all of which are invalid
    in valid UTF-8 text.  The converter assigns unused token IDs in
    the 0-255 range to maintain a bijective bytes_maps.
    """

    def test_roundtrip_ascii(self):
        """ASCII text should round-trip."""
        tok = Tokenizer.from_file("models/deepseek-llm.tbm")
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
        tok = Tokenizer.from_file("models/deepseek-llm.tbm")
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
        tok = Tokenizer.from_file("models/deepseek-llm.tbm")
        code = 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)'
        ids = tok.encode(code)
        decoded = tok.decode(ids)
        assert decoded == code

    def test_roundtrip_emoji(self):
        """Emoji should round-trip."""
        tok = Tokenizer.from_file("models/deepseek-llm.tbm")
        text = "👋😊🎉🔥💻"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_vocab_size(self):
        """Vocab size should be correct."""
        tok = Tokenizer.from_file("models/deepseek-llm.tbm")
        # DeepSeek-LLM has 100K total tokens
        assert 99000 <= tok.n_vocab <= 101000
        assert tok.n_vocab == 256 + len(tok.merges)

    def test_bytes_maps_bijection(self):
        """Byte mapping should be bijective (256 unique IDs)."""
        from tinybpe import load_model
        _, bm = load_model("models/deepseek-llm.tbm")
        assert bm is not None
        assert len(bm) == 256
        assert len(set(bm)) == 256, "bytes_maps must be bijective"

    def test_streaming_decode(self):
        """Streaming decode should match batch decode."""
        tok = Tokenizer.from_file("models/deepseek-llm.tbm")
        text = "hello world 你好 世界"
        ids = tok.encode(text)
        parts: list[str] = []
        decoder = tok.stream_decode(parts.append)
        for tid in ids:
            decoder(tid)
        assert "".join(parts) == text
