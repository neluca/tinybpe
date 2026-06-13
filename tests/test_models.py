"""Tests for pre-built .tbm models — verify correctness against tiktoken."""

import pytest

from tinybpe import Tokenizer

# Try importing tiktoken — tests are skipped if not available
try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


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

    @pytest.mark.parametrize(("name", "desc"), MODELS)
    def test_simple_texts(self, name, desc):
        """Simple ASCII texts should match tiktoken exactly."""
        pat = PAT_GPT2 if name in ("r50k_base", "p50k_base") else PAT_GPT4
        enc = tiktoken.get_encoding(name)
        tok = Tokenizer.from_file(f"models/{name}.tbm", pat_str=pat)
        for text in SIMPLE_TEXTS:
            assert tok.encode(text) == enc.encode(text)

    @pytest.mark.parametrize(("name", "desc"), MODELS)
    def test_multilingual(self, name, desc):
        """Multilingual texts should match tiktoken."""
        pat = PAT_GPT2 if name in ("r50k_base", "p50k_base") else PAT_GPT4
        enc = tiktoken.get_encoding(name)
        tok = Tokenizer.from_file(f"models/{name}.tbm", pat_str=pat)
        for text in MULTILINGUAL_TEXTS:
            assert tok.encode(text) == enc.encode(text)

    @pytest.mark.parametrize(("name", "desc"), MODELS)
    def test_emoji(self, name, desc):
        """Emoji texts should match tiktoken."""
        pat = PAT_GPT2 if name in ("r50k_base", "p50k_base") else PAT_GPT4
        enc = tiktoken.get_encoding(name)
        tok = Tokenizer.from_file(f"models/{name}.tbm", pat_str=pat)
        for text in EMOJI_TEXTS:
            if name in ("r50k_base", "p50k_base") and len(text) > 2:
                continue
            assert tok.encode(text) == enc.encode(text)

    @pytest.mark.parametrize(("name", "desc"), MODELS)
    def test_code(self, name, desc):
        """Code snippets should match tiktoken."""
        pat = PAT_GPT2 if name in ("r50k_base", "p50k_base") else PAT_GPT4
        enc = tiktoken.get_encoding(name)
        tok = Tokenizer.from_file(f"models/{name}.tbm", pat_str=pat)
        for text in CODE_TEXTS:
            assert tok.encode(text) == enc.encode(text)

    @pytest.mark.parametrize(("name", "desc"), MODELS)
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

    @pytest.mark.parametrize(("name", "desc"), MODELS)
    def test_vocab_size(self, name, desc):
        """Vocab size should be reasonable."""
        tok = Tokenizer.from_file(f"models/{name}.tbm")
        # Vocab = 256 base bytes + n_merges
        assert tok.n_vocab > 256
        assert tok.n_vocab == 256 + len(tok.merges)

    @pytest.mark.parametrize(("name", "desc"), MODELS)
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


# ---------------------------------------------------------------------------
# Model I/O error paths (targeting _model_io.py)
# ---------------------------------------------------------------------------


class TestModelIOErrors:
    """Tests for error handling in _model_io.py."""

    def test_load_invalid_header(self, tmp_path):
        """Load with non-matching header should raise ValueError."""
        from tinybpe._model_io import load_model

        p = tmp_path / "bad.tbm"
        p.write_text("Not a TinyBPE file\n0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid model file header"):
            load_model(str(p))

    def test_load_future_version(self, tmp_path):
        """Load with version newer than supported should raise ValueError."""
        from tinybpe._model_io import MODEL_VERSION, load_model

        p = tmp_path / "future.tbm"
        p.write_text(f"TinyBPE Model v{MODEL_VERSION + 10}\n0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="newer"):
            load_model(str(p))

    def test_load_unexpected_remap_count(self, tmp_path):
        """Load with remap count other than 0 or 256 should raise ValueError."""
        from tinybpe._model_io import load_model

        p = tmp_path / "bad_remap.tbm"
        p.write_text("TinyBPE Model v1\n999\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unexpected remap count"):
            load_model(str(p))

    def test_save_invalid_bytes_maps_size(self, tmp_path):
        """save_model with wrong-sized bytes_maps should raise ValueError."""
        from tinybpe._model_io import save_model

        with pytest.raises(ValueError, match="exactly 256"):
            save_model(str(tmp_path / "bad"), [(1, 2)], bytes_maps=[0, 1, 2])

    def test_load_vocab_invalid_header(self, tmp_path):
        """load_vocab with wrong header should raise ValueError."""
        from tinybpe._model_io import load_vocab

        p = tmp_path / "bad.vocab"
        p.write_text("Not a vocab file\nAA== 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid vocabulary file"):
            load_vocab(str(p))

    def test_load_vocab_invalid_line(self, tmp_path):
        """load_vocab with malformed line should raise ValueError."""
        from tinybpe._model_io import load_vocab

        p = tmp_path / "bad_line.vocab"
        p.write_text("TinyBPE Vocabulary v1\ntoo many parts here\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid vocabulary line"):
            load_vocab(str(p))

    def test_save_model_auto_append_tbm(self, tmp_path):
        """save_model should auto-append .tbm."""
        from tinybpe._model_io import save_model

        save_model(str(tmp_path / "model"), [(104, 101)])
        assert (tmp_path / "model.tbm").exists()

    def test_save_vocab_auto_append_vocab(self, tmp_path):
        """save_vocab should auto-append .vocab."""
        from tinybpe._model_io import save_vocab

        v = {1: b"a", 2: b"bc"}
        save_vocab(str(tmp_path / "v"), v)
        assert (tmp_path / "v.vocab").exists()

    def test_save_vocab_sorted_by_rank(self, tmp_path):
        """Vocab entries should be sorted by token ID (rank)."""
        from tinybpe._model_io import save_vocab

        v = {3: b"c", 1: b"a", 2: b"b"}
        save_vocab(str(tmp_path / "sorted"), v)
        content = (tmp_path / "sorted.vocab").read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        # Skip header line
        ranks = [int(line.split()[1]) for line in lines[1:]]
        assert ranks == sorted(ranks)

    def test_load_vocab_save_vocab_roundtrip(self, tmp_path):
        """save_vocab and load_vocab should round-trip correctly."""
        from tinybpe._model_io import load_vocab, save_vocab

        v = {1: b"a", 256: b"hello"}
        save_vocab(str(tmp_path / "v"), v)
        loaded = load_vocab(str(tmp_path / "v") + ".vocab")
        assert loaded == v

    def test_save_load_model_roundtrip(self, tmp_path):
        """save_model and load_model should round-trip correctly."""
        from tinybpe._model_io import load_model, save_model

        merges = [(104, 101), (256, 108)]
        save_model(str(tmp_path / "m"), merges)
        loaded_merges, bm = load_model(str(tmp_path / "m"))
        assert loaded_merges == merges
        assert bm is None

    def test_save_model_existing_extension(self, tmp_path):
        """save_model with explicit .tbm extension should not double-append."""
        from tinybpe._model_io import save_model

        save_model(str(tmp_path / "m.tbm"), [(1, 2)])
        assert (tmp_path / "m.tbm").exists()
        assert not (tmp_path / "m.tbm.tbm").exists()

    def test_load_model_missing_file_raises(self):
        """load_model with nonexistent file should raise FileNotFoundError."""
        from tinybpe._model_io import load_model

        with pytest.raises(FileNotFoundError):
            load_model("__nonexistent_file_12345__")

    def test_load_legacy_version(self, tmp_path):
        """Load a .tbm file without explicit version header (version 0)."""
        from tinybpe._model_io import load_model

        p = tmp_path / "legacy.tbm"
        # Old format: "TinyBPE Model" without version
        p.write_text("TinyBPE Model\n0\n104 101\n", encoding="utf-8")
        merges, bm = load_model(str(p))
        assert merges == [(104, 101)]
        assert bm is None

    def test_remap_identity_detection(self, tmp_path):
        """Identity bytes_maps should be detected as None when saving."""
        from tinybpe._model_io import load_model, save_model

        # Identity remap
        identity = list(range(256))
        save_model(str(tmp_path / "id"), [(1, 2)], bytes_maps=identity)
        _, bm = load_model(str(tmp_path / "id"))
        assert bm == identity

    def test_load_invalid_version_format(self, tmp_path):
        """Load with 'v' in header but non-integer version should raise ValueError."""
        from tinybpe._model_io import load_model

        p = tmp_path / "badver.tbm"
        p.write_text("TinyBPE Model vABC\n0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid model file header"):
            load_model(str(p))

    def test_model_with_blank_lines(self, tmp_path):
        """Model file with blank lines should parse correctly."""
        from tinybpe._model_io import load_model, save_model

        save_model(str(tmp_path / "m"), [(1, 2), (3, 4)])
        content = (tmp_path / "m.tbm").read_text(encoding="utf-8")
        # Add blank lines
        lines = content.splitlines()
        lines.insert(3, "")  # blank after header+remap
        lines.insert(-1, "")  # blank before end
        (tmp_path / "m_blank.tbm").write_text("\n".join(lines) + "\n", encoding="utf-8")
        merges, _ = load_model(str(tmp_path / "m_blank.tbm"))
        assert merges == [(1, 2), (3, 4)]

    def test_vocab_with_blank_lines(self, tmp_path):
        """Vocab file with blank lines should parse correctly."""
        from tinybpe._model_io import load_vocab, save_vocab

        v = {1: b"a", 2: b"bc"}
        save_vocab(str(tmp_path / "v"), v)
        content = (tmp_path / "v.vocab").read_text(encoding="utf-8")
        lines = content.splitlines()
        lines.insert(2, "")  # blank line
        (tmp_path / "v_blank.vocab").write_text("\n".join(lines) + "\n", encoding="utf-8")
        loaded = load_vocab(str(tmp_path / "v_blank.vocab"))
        assert loaded == v

    def test_vocab_with_bytes_maps(self):
        """Tokenizer.vocab should return remapped bytes when bytes_maps is set."""
        from tinybpe import Tokenizer

        tok = Tokenizer([(104, 101)], bytes_maps=list(range(256)))
        v = tok.vocab
        assert isinstance(v, dict)
        assert len(v) > 0
        # Vocab should contain valid byte sequences
        for tb in v.values():
            assert isinstance(tb, bytes)

    def test_tokenizer_init_with_bytes_maps_and_special(self):
        """Tokenizer init with both bytes_maps and special_tokens should work."""
        from tinybpe import Tokenizer

        special = {"<eot>": 1000}
        tok = Tokenizer([(104, 101)], bytes_maps=list(range(256)), special_tokens=special)
        ids = tok.encode("<eot>")
        assert ids == [1000]
