import unittest
import torch
import torch.testing as te
import han.vocabulary as v


class Test(unittest.TestCase):
    def test(self):
        sut = v.Vocabulary()
        sut.build(["You can now install TorchText using pip!"])

        res = sut.forward(["You not can", "install using"])

        te.assert_close(
            res,
            torch.Tensor(
                [
                    [sut["you"], sut.pad_id, sut["can"]],
                    [sut["install"], sut["using"], sut.pad_id],
                ]
            ),
        )
