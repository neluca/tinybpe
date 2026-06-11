"""Integration tests for the TinyBPE Tokenizer."""

from pathlib import Path

from tinybpe import Tokenizer, load_model, save_model, save_vocab, load_vocab

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
