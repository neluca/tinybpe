"""Tests for streaming decode edge cases."""

from __future__ import annotations

import pytest

from tinybpe import Tokenizer

# A simple merge list that builds tokens from bytes
_SIMPLE_MERGES = [(104, 101), (256, 108), (257, 108), (258, 111)]  # h+e=he, he+l=hel, hel+l=hell, hell+o=hello


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """Create a simple trained tokenizer for streaming tests."""
    return Tokenizer(_SIMPLE_MERGES, bytes_maps=None)


@pytest.fixture(scope="module")
def tokenizer_with_remap() -> Tokenizer:
    """Create a tokenizer with identity byte remapping."""
    return Tokenizer(_SIMPLE_MERGES, bytes_maps=list(range(256)))


class TestStreamingDecode:
    """Tests for streaming decode functionality."""

    def test_single_byte_tokens(self, tokenizer: Tokenizer) -> None:
        """Each single-byte token should produce one callback."""
        # Encode a simple string that won't merge
        ids = tokenizer.encode("xyz")
        parts: list[str] = []
        decoder = tokenizer.stream_decode(parts.append)
        for tid in ids:
            decoder(tid)
        assert "".join(parts) == "xyz"

    def test_multibyte_token(self, tokenizer: Tokenizer) -> None:
        """Multi-byte tokens should produce the complete text in one callback."""
        ids = tokenizer.encode("hello")
        # "hello" should merge into a single token
        parts: list[str] = []
        decoder = tokenizer.stream_decode(parts.append)
        for tid in ids:
            decoder(tid)
        assert "".join(parts) == "hello"

    def test_empty_stream(self, tokenizer: Tokenizer) -> None:
        """Streaming decode with empty token list should not call callback."""
        parts: list[str] = []
        tokenizer.stream_decode(parts.append)
        # Don't call decoder at all
        assert parts == []

    def test_stream_decode_reset(self, tokenizer: Tokenizer) -> None:
        """Reset should clear the internal cache."""
        # Encode a multi-byte character
        ids = tokenizer.encode("你")  # 你
        parts: list[str] = []
        decoder = tokenizer.stream_decode(parts.append)
        # Feed partial data then reset
        for tid in ids[:1]:  # partial
            decoder(tid)
        tokenizer.stream_decode_reset()
        # Should be able to start fresh
        decoder2 = tokenizer.stream_decode(parts.append)
        for tid in ids:
            decoder2(tid)
        assert "".join(parts) == "你"

    def test_byte_remap_streaming(self, tokenizer_with_remap: Tokenizer) -> None:
        """Streaming decode should work with identity byte remapping."""
        tok = tokenizer_with_remap
        ids = tok.encode("test")
        parts: list[str] = []
        decoder = tok.stream_decode(parts.append)
        for tid in ids:
            decoder(tid)
        assert "".join(parts) == "test"

    def test_unicode_multibyte_boundary(self, tokenizer: Tokenizer) -> None:
        """Unicode characters split across multiple tokens should reassemble."""
        ids = tokenizer.encode("café")  # cafe + combining accent
        parts: list[str] = []
        decoder = tokenizer.stream_decode(parts.append)
        for tid in ids:
            decoder(tid)
        assert "".join(parts) == "café"

    def test_cjk_characters(self, tokenizer: Tokenizer) -> None:
        """CJK characters should stream correctly."""
        ids = tokenizer.encode("你好")  # 你好
        parts: list[str] = []
        decoder = tokenizer.stream_decode(parts.append)
        for tid in ids:
            decoder(tid)
        assert "".join(parts) == "你好"


class TestModelStreaming:
    """Streaming tests with built-in models."""

    @pytest.mark.parametrize("model_name", ["cl100k_base", "r50k_base"])
    def test_model_streaming_ascii(self, model_name: str) -> None:
        """Streaming decode should match batch decode for built-in models."""
        tok = Tokenizer.from_pretrained(model_name)
        text = "hello world test testing"
        ids = tok.encode(text)
        parts: list[str] = []
        decoder = tok.stream_decode(parts.append)
        for tid in ids:
            decoder(tid)
        assert "".join(parts) == text

    @pytest.mark.parametrize("model_name", ["cl100k_base", "qwen25"])
    def test_model_streaming_cjk(self, model_name: str) -> None:
        """Streaming decode should work for CJK with built-in models."""
        tok = Tokenizer.from_pretrained(model_name)
        text = "你好世界"
        ids = tok.encode(text)
        parts: list[str] = []
        decoder = tok.stream_decode(parts.append)
        for tid in ids:
            decoder(tid)
        assert "".join(parts) == text
