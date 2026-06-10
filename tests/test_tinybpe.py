import tinybpe as tb
import unittest
from pathlib import Path
import tiktoken
import regex as re

file_cl100k_base = str(Path(__file__).parent.absolute().joinpath("cl100k_base.tinymodel"))
file_t_cl100k_base = str(Path(__file__).parent.absolute().joinpath("t_cl100k_base"))
file_regex = str(Path(__file__).parent.absolute().joinpath("regex.tinymodel"))
file_simple = str(Path(__file__).parent.absolute().joinpath("simple.tinymodel"))
file_simple_t_model = str(Path(__file__).parent.absolute().joinpath("t_simple"))
file_simple_chinese = str(Path(__file__).parent.absolute().joinpath("simple-chinese.tinymodel"))
file_simple_vocab = str(Path(__file__).parent.absolute().joinpath("simple.vocab"))
file_simple_t_vocab = str(Path(__file__).parent.absolute().joinpath("t_simple"))
file_text = str(Path(__file__).parent.absolute().joinpath("the-old-man-and-the-sea.txt"))

SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


class TestTinyBPE(unittest.TestCase):

    def test_simple(self):
        text = open(file_text, "r", encoding="utf-8").read()
        trainer_1 = tb.SimpleTrainer(text)
        vocab_size = 1000
        merges_size = vocab_size - 256
        for _ in range(merges_size):
            trainer_1.step()

        merges_1 = trainer_1.merges
        model = tb.load_bpe_model(file_simple)
        self.assertEqual(merges_1, model.merges)
        self.assertIsNone(model.bytes_maps)
        self.assertEqual(trainer_1.n_merges, merges_size)

        trainer_1.save(file_simple_t_model)
        file_simple_t_model_ = file_simple_t_model + ".tinymodel"
        text_model = open(file_simple, "r", encoding="utf-8").read()
        text_t_model = open(file_simple_t_model_, "r", encoding="utf-8").read()
        self.assertEqual(text_model, text_t_model)

        tokenizer = tb.CommonTokenizer(model.merges)
        s1 = "hello world, old man !"
        s2 = tokenizer.decode(tokenizer.encode(s1))
        self.assertEqual(s1, s2)
        s1 = "你好世界 1234"
        s2 = tokenizer.decode(tokenizer.encode(s1))
        self.assertEqual(s1, s2)
        self.assertEqual(tokenizer.n_vocab, vocab_size)
        self.assertEqual(tokenizer.merges, model.merges)

        tokenizer = tb.Tokenizer(model)
        s1 = "hello world, old man !"
        s2 = tokenizer.decode(tokenizer.encode(s1))
        self.assertEqual(s1, s2)
        s1 = "你好世界 1234"
        s2 = tokenizer.decode(tokenizer.encode(s1))
        self.assertEqual(s1, s2)
        self.assertEqual(tokenizer.n_vocab, vocab_size)
        self.assertEqual(tokenizer.merges, model.merges)

    def test_regex_train(self):
        class Preprocess:
            def __init__(self):
                self.pattern = SPLIT_PATTERN
                self.compiled_pattern = re.compile(self.pattern)

            def __call__(self, text_):
                text_chunks = re.findall(self.compiled_pattern, text_)
                ids = [ch.encode("utf-8") for ch in text_chunks]
                return ids

        proc = Preprocess()
        text = open(file_text, "r", encoding="utf-8").read()
        trainer = tb.SimpleTrainer(text, proc)
        vocab_size = 1000
        merges_size = vocab_size - 256
        for _ in range(merges_size):
            trainer.step()
        model = tb.load_bpe_model(file_regex)
        self.assertEqual(trainer.merges, model.merges)

    def test_regex(self):
        vocab_size = 1000
        special_tokens = {
            "<eot>": vocab_size,
            "<fim_prefix>": vocab_size + 1,
            "<fim_middle>": vocab_size + 2,
            "<fim_suffix>": vocab_size + 3,
            "<eop>": vocab_size + 4,
        }
        model = tb.load_bpe_model(file_regex)
        tokenizer = tb.Tokenizer(model, SPLIT_PATTERN, special_tokens=special_tokens)
        s1 = "<fim_prefix> hello world 中文<eot><fim_suffix>"
        ids = tokenizer.encode(s1)
        s2 = tokenizer.decode(ids)
        self.assertEqual(s1, s2)
        s1 = "你好世界 1234"
        s2 = tokenizer.decode(tokenizer.encode(s1))
        self.assertEqual(s1, s2)
        s1 = "<fim_prefix>"
        ids = tokenizer.encode(s1)
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], vocab_size + 1)
        s2 = tokenizer.decode(ids)
        self.assertEqual(s1, s2)

    def test_simple_chinese(self):
        model = tb.load_bpe_model(file_simple_chinese)
        tokenizer = tb.Tokenizer(model)
        s1 = "他是一个独自一人划着小船在墨西哥湾大海流打鱼的老人"
        s2 = tokenizer.decode(tokenizer.encode(s1))
        self.assertEqual(s1, s2)
        s1 = "👋😊🍍"
        s2 = tokenizer.decode(tokenizer.encode(s1))
        self.assertEqual(s1, s2)

    def test_vocab(self):
        model = tb.load_bpe_model(file_simple)
        tokenizer = tb.Tokenizer(model)
        self.assertEqual(tokenizer.n_vocab, 1000)
        tokenizer.save_vocab(file_simple_t_vocab)
        file_simple_t_vocab_ = file_simple_t_vocab + ".vocab"
        text_vocab = open(file_simple_vocab, "r", encoding="utf-8").read()
        text_t_vocab = open(file_simple_t_vocab_, "r", encoding="utf-8").read()
        self.assertEqual(text_vocab, text_t_vocab)

    def test_simple_stream_decode(self):
        model_simple = tb.load_bpe_model(file_simple)
        tokenizer_simple = tb.Tokenizer(model_simple)
        tokenizer_simple.stream_decode_cache_clean()
        s = "Hello world this is test case"
        ids = tokenizer_simple.encode(s)
        s1 = ""

        def cb(text):
            nonlocal s1
            s1 += text

        decode = tokenizer_simple.stream_decode(cb)
        for i in ids:
            decode(i)
        self.assertEqual(s, s1)


class TestFromTikToken(unittest.TestCase):

    def test_encode_encode(self):
        enc = tiktoken.get_encoding("cl100k_base")
        model = tb.get_from_tiktoken(enc._mergeable_ranks)
        tokenizer = tb.Tokenizer(model, SPLIT_PATTERN, special_tokens=enc._special_tokens)

        s1 = "Hello world this is test case. <|endoftext|>"
        tiktoken_ids = enc.encode(s1, allowed_special="all")
        ids = tokenizer.encode(s1)
        self.assertEqual(tiktoken_ids, ids)
        s2 = enc.decode(ids)
        s3 = tokenizer.decode(ids)
        self.assertEqual(s2, s3)

        s1 = "他是一个独自一人划着小船在墨西哥湾大海流打鱼的老人"
        tiktoken_ids = enc.encode(s1, allowed_special="all")
        ids = tokenizer.encode(s1)
        self.assertEqual(tiktoken_ids, ids)
        s2 = enc.decode(ids)
        s3 = tokenizer.decode(ids)
        self.assertEqual(s2, s3)

        s1 = "👋😉🏔️⛲⛈️☂️"
        tiktoken_ids = enc.encode(s1, allowed_special="all")
        ids = tokenizer.encode(s1)
        self.assertEqual(tiktoken_ids, ids)
        s2 = enc.decode(ids)
        s3 = tokenizer.decode(ids)
        self.assertEqual(s2, s3)

    def test_stream_decode(self):
        model = tb.load_bpe_model(file_cl100k_base)
        tokenizer = tb.Tokenizer(model)
        tokenizer.stream_decode_cache_clean()
        s = "Hello world this is test case. <|endoftext|>"
        ids = tokenizer.encode(s)
        s1 = ""

        def cb(text):
            nonlocal s1
            s1 += text

        decode = tokenizer.stream_decode(cb)
        for i in ids:
            decode(i)
        self.assertEqual(s, s1)

        s = "只是一句中文，你好世界"
        ids = tokenizer.encode(s)
        s1 = ""
        decode = tokenizer.stream_decode(cb)
        for i in ids:
            decode(i)
        self.assertEqual(s, s1)

        s = "👋😉🏔️⛲⛈️☂️"
        ids = tokenizer.encode(s)
        s1 = ""
        decode = tokenizer.stream_decode(cb)
        for i in ids:
            decode(i)
        self.assertEqual(s, s1)

    def test_save_load(self):
        enc = tiktoken.get_encoding("cl100k_base")
        tb.save_from_tiktoken(file_t_cl100k_base, enc._mergeable_ranks)
        file_t_cl100k_base_ = file_t_cl100k_base + ".tinymodel"
        text_model = open(file_cl100k_base, "r", encoding="utf-8").read()
        text_t_model = open(file_t_cl100k_base_, "r", encoding="utf-8").read()
        self.assertEqual(text_model, text_t_model)

        model = tb.load_bpe_model(file_cl100k_base)
        model2 = tb.get_from_tiktoken(enc._mergeable_ranks)
        self.assertEqual(model.merges, model2.merges)
