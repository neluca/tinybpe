import unittest
from tinybpe import bpe, Tokenizer


class TestBPETokenizer(unittest.TestCase):
    def test_encode(self):
        pass


class TestBPETrainer(unittest.TestCase):
    def test_train(self):
        trainer = bpe.Trainer([b"Hello TinyBPE", b"1234567890"])


class TestBPEBytesRemap(unittest.TestCase):
    def test_bytes_rmap(self):
        pass


class TestTokenizer(unittest.TestCase):
    def test_encode(self):
        pass


class TestSimpleTrainer(unittest.TestCase):
    def test_train(self):
        pass
