import unittest
import han.token as t


class Test(unittest.TestCase):
    def test(self):
        tokenizer = t.Tokenizer()
        res = tokenizer("You can now install TorchText using pip!")
        self.assertEqual(
            ["you", "can", "now", "install", "torchtext", "using", "pip", "!"],
            res,
        )
