"""Integration tests for the TinyBPE Tokenizer."""

from pathlib import Path

from tinybpe import Tokenizer, load_model, load_vocab, save_model, save_vocab

TESTS_DIR = Path(__file__).parent
FILE_SIMPLE = str(TESTS_DIR / "simple")
FILE_SIMPLE_CHINESE = str(TESTS_DIR / "simple-chinese")


class TestTokenizer:
    """Tests for the Tokenizer class (encode, decode, round-trip)."""

    def test_roundtrip_ascii(self):
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        text = "hello world, old man!"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_roundtrip_cjk(self):
        tok = Tokenizer.from_file(FILE_SIMPLE_CHINESE + ".tbm")
        text = "你好世界 1234"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_roundtrip_emoji(self):
        tok = Tokenizer.from_file(FILE_SIMPLE_CHINESE + ".tbm")
        text = "👋😊🍍"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_encode_ordinary(self):
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        text = "hello world"
        ids1 = tok.encode(text)
        ids2 = tok.encode_ordinary(text)
        assert ids1 == ids2  # no special tokens → same result

    def test_properties(self):
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        assert isinstance(tok.merges, list)
        assert len(tok.merges) > 0
        assert isinstance(tok.vocab, dict)
        assert tok.n_vocab == len(tok.vocab)
        assert tok.n_vocab == 256 + len(tok.merges)

    def test_save_load_model(self):
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        save_model(str(TESTS_DIR / "t_simple"), tok.merges)
        merges2, bm = load_model(str(TESTS_DIR / "t_simple") + ".tbm")
        assert tok.merges == merges2
        assert bm is None

    def test_save_load_vocab(self):
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        save_vocab(str(TESTS_DIR / "t_simple"), tok.vocab)
        vocab2 = load_vocab(str(TESTS_DIR / "t_simple") + ".vocab")
        assert tok.vocab == vocab2


class TestTokenizerSpecialTokens:
    """Tests for special token handling."""

    def test_special_token_roundtrip(self):
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        # Small vocab with special tokens
        special_tokens = {
            "<eot>": 1000,
            "<fim_prefix>": 1001,
            "<fim_suffix>": 1002,
        }
        tok2 = Tokenizer(
            tok.merges,
            special_tokens=special_tokens,
        )
        text = "<fim_prefix> hello world <eot> <fim_suffix>"
        ids = tok2.encode(text)
        decoded = tok2.decode(ids)
        assert decoded == text

    def test_special_token_single(self):
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        special_tokens = {"<eot>": 1000}
        tok2 = Tokenizer(tok.merges, special_tokens=special_tokens)
        ids = tok2.encode("<eot>")
        assert ids == [1000]
        assert tok2.decode(ids) == "<eot>"

    def test_special_vocab_size(self):
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        n_base = tok.n_vocab
        special_tokens = {"<a>": n_base, "<b>": n_base + 1}
        tok2 = Tokenizer(tok.merges, special_tokens=special_tokens)
        assert tok2.n_vocab == n_base + 2


class TestTokenizerStreaming:
    """Tests for streaming decode."""

    def test_stream_decode_ascii(self):
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        text = "Hello world this is a test case"
        ids = tok.encode(text)
        parts: list[str] = []

        decoder = tok.stream_decode(lambda s: parts.append(s))
        for tid in ids:
            decoder(tid)

        assert "".join(parts) == text

    def test_stream_decode_cjk(self):
        tok = Tokenizer.from_file(FILE_SIMPLE_CHINESE + ".tbm")
        text = "只是一句中文，你好世界"
        ids = tok.encode(text)
        parts: list[str] = []

        decoder = tok.stream_decode(lambda s: parts.append(s))
        for tid in ids:
            decoder(tid)

        assert "".join(parts) == text

    def test_stream_decode_emoji(self):
        tok = Tokenizer.from_file(FILE_SIMPLE_CHINESE + ".tbm")
        text = "👋😉🏔️⛲"
        ids = tok.encode(text)
        parts: list[str] = []

        decoder = tok.stream_decode(lambda s: parts.append(s))
        for tid in ids:
            decoder(tid)

        assert "".join(parts) == text

    def test_stream_decode_reset(self):
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        text = "hello"
        ids = tok.encode(text)
        parts: list[str] = []

        decoder = tok.stream_decode(lambda s: parts.append(s))
        for tid in ids:
            decoder(tid)
        assert "".join(parts) == text

        # Reset and decode again
        tok.stream_decode_reset()
        parts2: list[str] = []
        decoder2 = tok.stream_decode(lambda s: parts2.append(s))
        for tid in ids:
            decoder2(tid)
        assert "".join(parts2) == text


class TestTokenizerWithPattern:
    """Tests for regex pre-tokenization."""

    def test_regex_pattern(self):
        tok = Tokenizer.from_file(
            FILE_SIMPLE + ".tbm",
            pat_str=r"\w+|\s+|[^\w\s]+",
        )
        text = "hello world, hi!"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text


class TestTokenizerLoadModel:
    """Tests for model loading edge cases."""

    def test_missing_extension(self):
        # load_model now auto-appends .tbm — nonexistent file raises FileNotFoundError
        with __import__("pytest").raises(FileNotFoundError):
            load_model("no_extension")

    def test_nonexistent_file(self):
        with __import__("pytest").raises(FileNotFoundError):
            load_model("nonexistent.tbm")


class TestTokenizerRepr:
    """Tests for __repr__()."""

    def test_repr_no_remap_no_special(self):
        """__repr__ should show vocab size and remap=False when no remap."""
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        r = repr(tok)
        assert "n_vocab=" in r
        assert "byte_remap=False" in r
        assert "special_tokens=0" in r

    def test_repr_with_remap(self):
        """__repr__ should show byte_remap=True when bytes_maps is set."""
        tok = Tokenizer([(104, 101)], bytes_maps=list(range(256)))
        r = repr(tok)
        assert "byte_remap=True" in r

    def test_repr_with_special(self):
        """__repr__ should show special_tokens count."""
        tok = Tokenizer([(104, 101)], special_tokens={"<eot>": 1000})
        r = repr(tok)
        assert "special_tokens=1" in r

    def test_repr_multiple_special(self):
        """__repr__ should show correct special token count."""
        special = {"<a>": 1000, "<b>": 1001, "<c>": 1002}
        tok = Tokenizer([(104, 101)], special_tokens=special)
        r = repr(tok)
        assert "special_tokens=3" in r


class TestTokenizerSave:
    """Tests for Tokenizer.save() and save_vocab() instance methods."""

    def test_save_auto_append_extension(self, tmp_path):
        """Tokenizer.save() should auto-append .tbm."""
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        out = str(tmp_path / "out_no_ext")
        tok.save(out)
        assert (tmp_path / "out_no_ext.tbm").exists()

    def test_save_vocab_auto_append_extension(self, tmp_path):
        """Tokenizer.save_vocab() should auto-append .vocab."""
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        out = str(tmp_path / "v_no_ext")
        tok.save_vocab(out)
        assert (tmp_path / "v_no_ext.vocab").exists()

    def test_save_with_existing_extension(self, tmp_path):
        """Tokenizer.save() should work with explicit .tbm extension."""
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        out = str(tmp_path / "out.tbm")
        tok.save(out)
        assert (tmp_path / "out.tbm").exists()
        # Should not create double extension
        assert not (tmp_path / "out.tbm.tbm").exists()

    def test_save_vocab_with_existing_extension(self, tmp_path):
        """Tokenizer.save_vocab() should work with explicit .vocab."""
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm")
        out = str(tmp_path / "v.vocab")
        tok.save_vocab(out)
        assert (tmp_path / "v.vocab").exists()


class TestTokenizerSpecialEdgeCases:
    """Edge case tests for special token handling."""

    def test_encode_only_special(self):
        """Text consisting entirely of a special token should encode as that ID."""
        tok = Tokenizer([(104, 101)], special_tokens={"<eot>": 1000})
        assert tok.encode("<eot>") == [1000]

    def test_encode_special_overlap(self):
        """When one special is a prefix of another, the longer should match."""
        special = {"<a>": 1000, "<ab>": 1001}
        tok = Tokenizer([(104, 101)], special_tokens=special)
        ids = tok.encode("<ab>")
        assert ids == [1001]

    def test_encode_mixed_special_and_text(self):
        """Special tokens interspersed with text should encode correctly."""
        special = {"<eot>": 1000, "<bos>": 1001}
        tok = Tokenizer([(104, 101)], special_tokens=special)
        ids = tok.encode("<bos>hello<eot>")
        assert ids[0] == 1001  # bos
        assert ids[-1] == 1000  # eot

    def test_special_tokens_with_encode_and_encode_ordinary(self):
        """Both encode and encode_ordinary should handle special tokens correctly."""
        special = {"<eot>": 1000}
        tok = Tokenizer([(104, 101)], special_tokens=special)
        # encode with special tokens
        regular_ids = tok.encode("<eot>")
        assert regular_ids == [1000]
        assert tok.decode(regular_ids) == "<eot>"
        # encode_ordinary also passes special tokens to C engine
        ordinary_ids = tok.encode_ordinary("<eot>")
        # Both should produce the same result for token-only text
        assert ordinary_ids == [1000]

    def test_special_with_byte_remap(self):
        """Special tokens should work with bytes_maps enabled."""
        special = {"<eot>": 1000}
        tok = Tokenizer([(104, 101)], bytes_maps=list(range(256)), special_tokens=special)
        ids = tok.encode("<eot>")
        assert ids == [1000]
        assert tok.decode(ids) == "<eot>"


class TestTokenizerFromFile:
    """Additional tests for Tokenizer.from_file()."""

    def test_from_file_auto_append_tbm(self):
        """from_file should auto-append .tbm when path has no extension."""
        tok = Tokenizer.from_file(FILE_SIMPLE)  # no .tbm
        assert tok.n_vocab > 0

    def test_from_file_with_pat_str(self):
        """from_file with pat_str should apply the pattern."""
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm", pat_str=r"\w+|\s+")
        text = "hello world hi"
        ids = tok.encode(text)
        assert tok.decode(ids) == text

    def test_from_file_with_special_tokens(self):
        """from_file with special_tokens should work correctly."""
        special = {"<eot>": 2000}
        tok = Tokenizer.from_file(FILE_SIMPLE + ".tbm", special_tokens=special)
        ids = tok.encode("<eot>")
        assert ids == [2000]
