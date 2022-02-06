import unittest
import torch.testing as te
import han.embedding as e


class Test(unittest.TestCase):
    def test(self):
        sut = e.Vocabulary()
        sut.build(["You can now install TorchText using pip!"])
        print(sut.forward(["You can not"]))

        assert False
