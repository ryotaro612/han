import unittest
import torchtext.vocab as v
import han.token as t


class Test(unittest.TestCase):
    def test(self):
        tokenizer = t.Tokenizer()
        res = tokenizer("You can now install TorchText using pip!")
        self.assertEqual(
            ["you", "can", "now", "install", "torchtext", "using", "pip", "!"],
            res,
        )

    def test_temp(self):
        a = v.build_vocab_from_iterator([["a", "be", "long", "vacation"]])
        print(a)

        print(a.get_stoi())
        print(a["be"])
        print(a["dogee"])
        assert False
