"""Tests for the built-in model registry and from_pretrained()."""

from __future__ import annotations

import pytest

from tinybpe import Tokenizer, list_models
from tinybpe._registry import _MODEL_REGISTRY, get_model_info


class TestListModels:
    """Tests for list_models()."""

    def test_returns_all_models(self) -> None:
        """All built-in models should be listed."""
        models = list_models()
        assert len(models) >= 8  # at least 8, grows as models are added

    def test_is_sorted(self) -> None:
        """Model names should be sorted alphabetically."""
        models = list_models()
        assert models == sorted(models)

    def test_contains_expected_names(self) -> None:
        """Expected model names should be present."""
        models = list_models()
        expected = {
            "cl100k_base",
            "o200k_base",
            "p50k_base",
            "r50k_base",
            "qwen35",
            "phi2",
            "deepseek-llm",
            "minicpm5",
        }
        assert expected.issubset(set(models))


class TestGetModelInfo:
    """Tests for get_model_info()."""

    def test_valid_model_returns_metadata(self) -> None:
        """get_model_info should return metadata for a valid model."""
        info = get_model_info("cl100k_base")
        assert info["name"] == "cl100k_base"
        assert info["path"] == "models/cl100k_base.tbm"
        assert info["vocab_size"] > 0
        assert info["family"] == "GPT-4"
        assert info["has_byte_remap"] is True

    def test_unknown_model_raises(self) -> None:
        """get_model_info should raise ValueError for unknown models."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_info("nonexistent")

    def test_all_models_have_required_fields(self) -> None:
        """Every registry entry should have all ModelInfo fields."""
        required_keys = {
            "name",
            "path",
            "vocab_size",
            "description",
            "family",
            "pat_str",
            "special_tokens",
            "has_byte_remap",
        }
        for name, info in _MODEL_REGISTRY.items():
            missing = required_keys - set(info.keys())
            assert not missing, f"Model {name!r} missing keys: {missing}"


class TestFromPretrained:
    """Tests for Tokenizer.from_pretrained()."""

    def test_unknown_model_raises(self) -> None:
        """from_pretrained should raise ValueError for unknown name."""
        with pytest.raises(ValueError, match="Unknown model"):
            Tokenizer.from_pretrained("nonexistent")

    def test_cl100k_base_loads(self) -> None:
        """cl100k_base model should load and encode text."""
        tok = Tokenizer.from_pretrained("cl100k_base")
        ids = tok.encode("hello world")
        assert len(ids) > 0
        assert tok.decode(ids) == "hello world"

    def test_o200k_base_loads(self) -> None:
        """o200k_base model should load."""
        tok = Tokenizer.from_pretrained("o200k_base")
        ids = tok.encode("hello world")
        assert len(ids) > 0
        assert tok.decode(ids) == "hello world"

    def test_p50k_base_loads(self) -> None:
        """p50k_base model should load."""
        tok = Tokenizer.from_pretrained("p50k_base")
        ids = tok.encode("hello world")
        assert len(ids) > 0
        assert tok.decode(ids) == "hello world"

    def test_r50k_base_loads(self) -> None:
        """r50k_base model should load."""
        tok = Tokenizer.from_pretrained("r50k_base")
        ids = tok.encode("hello world")
        assert len(ids) > 0
        assert tok.decode(ids) == "hello world"

    def test_qwen35_loads(self) -> None:
        """qwen35 model should load."""
        tok = Tokenizer.from_pretrained("qwen35")
        ids = tok.encode("hello world")
        assert len(ids) > 0
        assert tok.decode(ids) == "hello world"

    def test_phi2_loads(self) -> None:
        """phi2 model should load."""
        tok = Tokenizer.from_pretrained("phi2")
        ids = tok.encode("hello world")
        assert len(ids) > 0
        assert tok.decode(ids) == "hello world"

    def test_deepseek_llm_loads(self) -> None:
        """deepseek-llm model should load."""
        tok = Tokenizer.from_pretrained("deepseek-llm")
        ids = tok.encode("hello world")
        assert len(ids) > 0
        assert tok.decode(ids) == "hello world"

    def test_minicpm5_loads(self) -> None:
        """minicpm5 model should load and roundtrip."""
        tok = Tokenizer.from_pretrained("minicpm5")
        ids = tok.encode("hello world")
        assert len(ids) > 0
        assert tok.decode(ids) == "hello world"

    @pytest.mark.parametrize(
        "model_name",
        [
            "cl100k_base",
            "o200k_base",
            "p50k_base",
            "r50k_base",
        ],
    )
    def test_tiktoken_models_roundtrip_ascii(self, model_name: str) -> None:
        """TikToken-based models should roundtrip ASCII text."""
        tok = Tokenizer.from_pretrained(model_name)
        for text in ["hello world", "Hello World! How are you?", "test 123"]:
            ids = tok.encode(text)
            assert tok.decode(ids) == text

    @pytest.mark.parametrize(
        "model_name",
        [
            "cl100k_base",
            "o200k_base",
        ],
    )
    def test_tiktoken_models_roundtrip_unicode(self, model_name: str) -> None:
        """TikToken-based models should roundtrip Unicode text."""
        tok = Tokenizer.from_pretrained(model_name)
        for text in ["你好世界", "日本語テスト", "한국어", "Привет"]:
            ids = tok.encode(text)
            assert tok.decode(ids) == text

    def test_registry_paths_exist(self) -> None:
        """Every registry path should point to an existing file."""
        import os

        from tinybpe.tokenizer import _find_package_file

        for name, info in _MODEL_REGISTRY.items():
            path = info["path"]
            abs_path = _find_package_file(path)
            assert os.path.isfile(abs_path), f"Model {name!r}: {abs_path} not found"

    def test_vocab_size_consistency(self) -> None:
        """Registry vocab_size should match the loaded tokenizer's n_vocab."""
        for name in ["r50k_base", "p50k_base"]:
            info = _MODEL_REGISTRY[name]
            tok = Tokenizer.from_pretrained(name)
            # Allow tolerance for special tokens registered differently
            assert abs(tok.n_vocab - info["vocab_size"]) <= 2
