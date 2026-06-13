"""Tests for conversion script helper functions.

These test pure functions in scripts/ that don't require
external dependencies like tiktoken or huggingface_hub.
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Path to scripts directory
_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _import_script(name: str):
    """Import a script module by its file path."""
    path = os.path.join(_SCRIPTS_DIR, name)
    spec = importlib.util.spec_from_file_location(name.rstrip(".py"), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestConvertHFByteMapping:
    """Tests for GPT-2 ByteLevel mapping used in conversion scripts."""

    def test_import_convert_hf(self) -> None:
        """convert_hf_tokenizer.py should be importable."""
        mod = _import_script("convert_hf_tokenizer.py")
        assert hasattr(mod, "_bytes_to_unicode")
        assert hasattr(mod, "_detect_bytelevel")

    def test_bytes_to_unicode_is_bijection(self) -> None:
        """The GPT-2 byte-to-unicode mapping should be a bijection (256 entries)."""
        mod = _import_script("convert_hf_tokenizer.py")
        b2u = mod._bytes_to_unicode()
        assert len(b2u) == 256
        # All byte values (0-255) as keys
        assert set(b2u.keys()) == set(range(256))
        # All output values unique (bijection)
        assert len(set(b2u.values())) == 256

    def test_bytes_to_unicode_printable_identity(self) -> None:
        """Printable ASCII characters should map to themselves."""
        mod = _import_script("convert_hf_tokenizer.py")
        b2u = mod._bytes_to_unicode()
        for b in range(33, 127):  # ! to ~
            assert b2u[b] == b

    def test_bytes_to_unicode_roundtrip(self) -> None:
        """Unicode-to-byte should invert byte-to-unicode."""
        mod = _import_script("convert_hf_tokenizer.py")
        b2u = mod._bytes_to_unicode()
        u2b = mod._UNICODE_TO_BYTE
        for b in range(256):
            uc = b2u[b]
            assert u2b[uc] == b

    def test_detect_bytelevel_true(self) -> None:
        """_detect_bytelevel should return True for ByteLevel pre-tokenizer."""
        mod = _import_script("convert_hf_tokenizer.py")
        tokenizer_json = {"pre_tokenizer": {"type": "ByteLevel"}}
        assert mod._detect_bytelevel(tokenizer_json) is True

    def test_detect_bytelevel_from_decoder(self) -> None:
        """_detect_bytelevel should detect ByteLevel from decoder."""
        mod = _import_script("convert_hf_tokenizer.py")
        tokenizer_json = {"decoder": {"type": "ByteLevel"}}
        assert mod._detect_bytelevel(tokenizer_json) is True

    def test_detect_bytelevel_sequence(self) -> None:
        """_detect_bytelevel should handle nested Sequence pre-tokenizer."""
        mod = _import_script("convert_hf_tokenizer.py")
        tokenizer_json = {
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [{"type": "ByteLevel"}],
            }
        }
        assert mod._detect_bytelevel(tokenizer_json) is True

    def test_detect_bytelevel_false(self) -> None:
        """_detect_bytelevel should return False for non-ByteLevel tokenizer."""
        mod = _import_script("convert_hf_tokenizer.py")
        tokenizer_json = {"pre_tokenizer": {"type": "Metaspace"}}
        assert mod._detect_bytelevel(tokenizer_json) is False

    def test_detect_bytelevel_empty(self) -> None:
        """_detect_bytelevel should return False for empty config."""
        mod = _import_script("convert_hf_tokenizer.py")
        assert mod._detect_bytelevel({}) is False

    def test_detect_byte_mapping_full(self) -> None:
        """_detect_byte_mapping should find all 256 byte tokens in a full vocab."""
        mod = _import_script("convert_hf_tokenizer.py")
        b2u = mod._bytes_to_unicode()
        # Build a synthetic vocab with all byte tokens
        vocab = {}
        for b in range(256):
            ch = chr(b2u[b])
            vocab[ch] = b  # simple mapping: token ID = byte value
        result = mod._detect_byte_mapping(vocab)
        assert result is not None
        assert len(result) == 256
        # All byte values should map to themselves (identity in this case)
        assert all(result[i] == i for i in range(256))

    def test_detect_byte_mapping_incomplete(self) -> None:
        """_detect_byte_mapping should return None for incomplete vocab."""
        mod = _import_script("convert_hf_tokenizer.py")
        # Missing a required byte (0x20 = space)
        vocab = {"a": 0, "b": 1}  # far too few tokens
        result = mod._detect_byte_mapping(vocab)
        assert result is None


class TestConvertTikToken:
    """Smoke tests for convert_tiktoken.py."""

    def test_import_convert_tiktoken(self) -> None:
        """convert_tiktoken.py should be valid Python and contain expected imports."""
        path = os.path.join(_SCRIPTS_DIR, "convert_tiktoken.py")
        with open(path, encoding="utf-8") as f:
            source = f.read()
        # Should be valid Python
        compile(source, path, "exec")
        assert "decompose" in source or "mergeable" in source.lower()
