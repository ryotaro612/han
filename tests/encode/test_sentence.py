import unittest
import torchtext.vocab as v
import han.encode.sentence as s


class SentenceEncoderTestCase(unittest.TestCase):
    def test(self):
        vocab = v.build_vocab_from_iterator([["apple", "is", "tasty"]])

        sut = s.SentenceEncoder(vocab)

        res = sut.forward(["apple is tasty", "tasty is apple"])
        self.assertEqual(len(res), 2)
