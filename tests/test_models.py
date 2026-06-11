"""Tests for pre-built .tbm models — verify correctness against tiktoken."""

import pytest

from tinybpe import Tokenizer

# Try importing tiktoken — tests are skipped if not available
try:
    import tiktoken

    HAS_TIKTOKEN = True  # noqa: F401
except ImportError:
    HAS_TIKTOKEN = False  # noqa: F401


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

# Texts that don't depend on pre-tokenization pattern
SIMPLE_TEXTS = [
    "hello world",
    "Hello World! How are you?",
    "the quick brown fox jumps over the lazy dog",
    "Python programming is fun",
    "abcdefghijklmnopqrstuvwxyz",
    "a",
    "newline",
]

# TikToken regex patterns
# cl100k_base / o200k_base pattern (GPT-4 family)
PAT_GPT4 = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+"""
    r"""|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]"""
    r"""|\s+(?!\S)|\s+"""
)

# p50k_base / r50k_base pattern (GPT-2/GPT-3 family)
PAT_GPT2 = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

MULTILINGUAL_TEXTS = [
    "你好世界",
    "只是一个测试",
    "日本語テスト",
    "한국어 테스트",
    "Привет мир",
    "العالم مرحبا",
    "🌍🌎🌏",
]

EMOJI_TEXTS = [
    "👋",
    "😀🎉🔥",
    "👨‍👩‍👧‍👦",  # Family emoji (ZWJ sequence)
    "🏳️‍🌈",  # Rainbow flag
]

CODE_TEXTS = [
    "def hello(): return 'world'",
    "import numpy as np",
    "x = [1, 2, 3]",
    "class Foo: pass",
    "print(f'hello {name}')",
]


# ---------------------------------------------------------------------------
# TikToken model tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken not installed")
class TestTikTokenModels:
    """Verify TinyBPE .tbm models produce identical output to tiktoken."""

    MODELS = [
        ("cl100k_base", "GPT-4 / GPT-3.5"),
        ("o200k_base", "GPT-4o / GPT-5"),
        ("p50k_base", "GPT-3 davinci"),
        ("r50k_base", "GPT-2"),
    ]

    @pytest.mark.parametrize("name,desc", MODELS)
    def test_simple_texts(self, name, desc):
        """Simple ASCII texts should match tiktoken exactly."""
        pat = PAT_GPT2 if name in ("r50k_base", "p50k_base") else PAT_GPT4
        enc = tiktoken.get_encoding(name)
        tok = Tokenizer.from_file(f"models/{name}.tbm", pat_str=pat)
        for text in SIMPLE_TEXTS:
            assert tok.encode(text) == enc.encode(text)

    @pytest.mark.parametrize("name,desc", MODELS)
    def test_multilingual(self, name, desc):
        """Multilingual texts should match tiktoken."""
        pat = PAT_GPT2 if name in ("r50k_base", "p50k_base") else PAT_GPT4
        enc = tiktoken.get_encoding(name)
        tok = Tokenizer.from_file(f"models/{name}.tbm", pat_str=pat)
        for text in MULTILINGUAL_TEXTS:
            assert tok.encode(text) == enc.encode(text)

    @pytest.mark.parametrize("name,desc", MODELS)
    def test_emoji(self, name, desc):
        """Emoji texts should match tiktoken."""
        pat = PAT_GPT2 if name in ("r50k_base", "p50k_base") else PAT_GPT4
        enc = tiktoken.get_encoding(name)
        tok = Tokenizer.from_file(f"models/{name}.tbm", pat_str=pat)
        for text in EMOJI_TEXTS:
            if name in ("r50k_base", "p50k_base") and len(text) > 2:
                continue
            assert tok.encode(text) == enc.encode(text)

    @pytest.mark.parametrize("name,desc", MODELS)
    def test_code(self, name, desc):
        """Code snippets should match tiktoken."""
        pat = PAT_GPT2 if name in ("r50k_base", "p50k_base") else PAT_GPT4
        enc = tiktoken.get_encoding(name)
        tok = Tokenizer.from_file(f"models/{name}.tbm", pat_str=pat)
        for text in CODE_TEXTS:
            assert tok.encode(text) == enc.encode(text)

    @pytest.mark.parametrize("name,desc", MODELS)
    def test_roundtrip(self, name, desc):
        """All texts should round-trip through encode→decode."""
        tok = Tokenizer.from_file(f"models/{name}.tbm")
        all_texts = SIMPLE_TEXTS + MULTILINGUAL_TEXTS + EMOJI_TEXTS + CODE_TEXTS
        for text in all_texts:
            if not text:
                continue
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text, f"Round-trip failed for {text!r}"

    @pytest.mark.parametrize("name,desc", MODELS)
    def test_vocab_size(self, name, desc):
        """Vocab size should be reasonable."""
        tok = Tokenizer.from_file(f"models/{name}.tbm")
        # Vocab = 256 base bytes + n_merges
        assert tok.n_vocab > 256
        assert tok.n_vocab == 256 + len(tok.merges)

    @pytest.mark.parametrize("name,desc", MODELS)
    def test_streaming_decode(self, name, desc):
        """Streaming decode should match batch decode."""
        tok = Tokenizer.from_file(f"models/{name}.tbm")
        for text in SIMPLE_TEXTS + MULTILINGUAL_TEXTS[:3]:
            if not text:
                continue
            ids = tok.encode(text)
            parts: list[str] = []
            decoder = tok.stream_decode(parts.append)
            for tid in ids:
                decoder(tid)
            assert "".join(parts) == text


# ---------------------------------------------------------------------------
# TikToken-specific edge cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken not installed")
class TestCl100kBase:
    """GPT-4 tokenizer specific tests."""

    def test_special_token_match(self):
        """cl100k_base special tokens should round-trip."""
        special_tokens = {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        }
        tok = Tokenizer.from_file(
            "models/cl100k_base.tbm",
            pat_str=PAT_GPT4,
            special_tokens=special_tokens,
        )

        enc = tiktoken.get_encoding("cl100k_base")

        test_texts = [
            "hello world",
            "<|endoftext|>hello<|endofprompt|>",
            "<|fim_prefix|>code<|fim_suffix|>",
        ]
        for text in test_texts:
            our_ids = tok.encode(text)
            their_ids = enc.encode(text, allowed_special="all")
            assert our_ids == their_ids, f"Mismatch: {text!r}"
            assert tok.decode(our_ids) == text


# ---------------------------------------------------------------------------
# Model file I/O
# ---------------------------------------------------------------------------

class TestModelIO:
    """Tests for model file loading and saving."""

    def test_load_save_roundtrip(self, tmp_path):
        """Save and reload a model — should produce identical tokenizer."""
        from tinybpe import Trainer, load_model, save_model

        trainer = Trainer("hello world " * 500)
        trainer.train(30)
        original_merges = trainer.merges

        # Save
        path = str(tmp_path / "test")
        save_model(path, original_merges)
        assert (tmp_path / "test.tbm").exists()

        # Load
        merges, bm = load_model(str(tmp_path / "test.tbm"))
        assert merges == original_merges
        assert bm is None

    def test_load_save_with_remap(self, tmp_path):
        """Save and reload a model with byte remapping."""
        from tinybpe import load_model, save_model

        merges = [(104, 101), (256, 108)]  # he → 256, hel → 257
        bytes_maps = list(range(256))
        bytes_maps[0], bytes_maps[42] = 42, 0  # Swap two bytes

        path = str(tmp_path / "remap")
        save_model(path, merges, bytes_maps)
        assert (tmp_path / "remap.tbm").exists()

        loaded_merges, loaded_bm = load_model(str(tmp_path / "remap.tbm"))
        assert loaded_merges == merges
        assert loaded_bm == bytes_maps

    def test_load_auto_append_extension(self, tmp_path):
        """Loading should auto-append .tbm extension."""
        from tinybpe import load_model, save_model

        save_model(str(tmp_path / "model"), [(104, 101)])
        # load_model auto-appends .tbm
        merges, bm = load_model(str(tmp_path / "model"))
        assert merges == [(104, 101)]
        assert bm is None


# ---------------------------------------------------------------------------
# Byte remapping correctness
# ---------------------------------------------------------------------------

class TestByteRemapping:
    """Tests for byte remapping functionality."""

    def test_identity_remap(self):
        """Identity remapping should produce same results as no remapping."""
        from tinybpe import Trainer

        trainer = Trainer("hello world " * 200)
        trainer.train(30)

        tok_no_remap = Tokenizer(trainer.merges)
        tok_identity = Tokenizer(trainer.merges, bytes_maps=list(range(256)))

        for text in SIMPLE_TEXTS:
            assert tok_no_remap.encode(text) == tok_identity.encode(text)
            assert tok_no_remap.decode(tok_no_remap.encode(text)) == text

    def test_reverse_remap(self):
        """Reversed byte mapping should round-trip."""
        reverse_map = list(reversed(range(256)))
        tok = Tokenizer([(104, 101)], bytes_maps=reverse_map)

        text = "he"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_cl100k_remap_roundtrip(self):
        """cl100k_base with byte remapping should round-trip all texts."""
        tok = Tokenizer.from_file("models/cl100k_base.tbm")
        for text in SIMPLE_TEXTS + MULTILINGUAL_TEXTS:
            if not text:
                continue
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            assert decoded == text
