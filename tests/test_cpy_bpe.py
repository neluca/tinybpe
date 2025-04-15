from tinybpe import bpe
import unittest


class TestBPE(unittest.TestCase):
    def setUp(self) -> None:
        self.text1 = "hello world, hello python!"
        self.text2 = "你好世界，你好编程"
        self.merges = [(104, 101), (256, 108), (257, 108), (258, 111), (259, 32), (228, 189), (261, 160), (262, 229),
                       (263, 165), (264, 189), (150, 231), (260, 119)]
        self.special_tokens = {b"<eot>": 256 + len(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def test_trainer(self) -> None:
        trainer = bpe.Trainer([self.text1.encode("utf-8"), self.text2.encode("utf-8")])

        for i in range(11):
            pair, rank, freq = trainer.step()
            self.assertEqual(pair, self.merges[i])
            self.assertEqual(rank, 256 + i)
            self.assertEqual(freq, 2)

        merges_size = trainer.merges_size
        self.assertEqual(merges_size, 11)
        trainer.step()
        self.assertEqual(trainer.merges, self.merges)

    def test_continue_training(self) -> None:
        trainer = bpe.Trainer([self.text1.encode("utf-8"), self.text2.encode("utf-8")])
        trainer.load_merges(self.merges[:5])
        merges_size = trainer.merges_size
        self.assertEqual(merges_size, 5)

        for i in range(7):
            pair, rank, freq = trainer.step()
            self.assertEqual(pair, self.merges[i + 5])
            self.assertEqual(rank, 256 + 5 + i)

    def test_tokenizer(self) -> None:
        tokenizer_1 = bpe.Tokenizer(self.merges, self.special_tokens)
        tokenizer_2 = bpe.Tokenizer(self.merges)

        self.assertEqual(tokenizer_1.merges, self.merges)
        self.assertEqual(tokenizer_2.merges, self.merges)
        self.assertEqual(tokenizer_1.size, 256 + 13)
        self.assertEqual(tokenizer_2.size, 256 + 12)

        vocab_1 = tokenizer_1.vocab
        vocab_2 = tokenizer_2.vocab

        self.assertEqual(vocab_1[tokenizer_1.size - 1], b"<eot>")
        for i in range(tokenizer_2.size):
            self.assertEqual(vocab_1[i], vocab_2[i])

        for i in range(256):
            self.assertEqual(vocab_1[i], self.vocab[i])

    def test_encode_decode(self) -> None:
        tokenizer = bpe.Tokenizer(self.merges, self.special_tokens)
        s1 = b"hello, my friends"
        ids = tokenizer.encode(s1)
        s2 = tokenizer.decode(ids)
        self.assertEqual(s1, s2)

        special_id = tokenizer.encode(b"<eot>")
        self.assertEqual(special_id, [256 + len(self.merges)])

        ids = tokenizer.encode("你好".encode("utf-8"))
        self.assertEqual(len(ids), 1)

        s3 = tokenizer.decode(ids)
        self.assertEqual(s3, "你好".encode("utf-8"))

    def test_cache_decode(self) -> None:
        tokenizer = bpe.Tokenizer(self.merges, self.special_tokens)
        tokenizer.cache_clean()
        ids = tokenizer.encode("你好世界".encode("utf-8"))
        s = tokenizer.cache_decode(ids[0])
        self.assertEqual(s, "你好".encode("utf-8"))
        s = tokenizer.cache_decode(ids[1])
        self.assertIsNone(s)
        s = tokenizer.cache_decode(ids[2])
        self.assertIsNone(s)
        s = tokenizer.cache_decode(ids[3])
        self.assertEqual(s, "世".encode("utf-8"))
        s = tokenizer.cache_decode(ids[4])
        self.assertIsNone(s)
        s = tokenizer.cache_decode(ids[5])
        self.assertEqual(s, "界".encode("utf-8"))

    def test_bytes_remap(self) -> None:
        r1 = [i for i in range(256)]
        r2 = [i for i in reversed(r1)]
        bytes_remap_1 = bpe.BytesRemap(r1)
        bytes_remap_2 = bpe.BytesRemap(r2)
        s1 = b"abcdef"
        s2 = bytes_remap_1(s1)
        s3 = bytes_remap_2(s1)
        self.assertEqual(s1, s2)
        self.assertEqual(s3, b'\x9e\x9d\x9c\x9b\x9a\x99')
