import unittest
import torch
import torch.testing as te
import han.model.attention as a


class AttentionModelTestCase(unittest.TestCase):
    def test_calc_args_softmax(self):
        sut = a.AttentionModel(1)
        u = torch.Tensor(
            [
                [[1, 1, 1], [2, 3, 4]],
                [[1, 2, 2], [1, 3, 2]],
                [[1, 0, 1], [2, 1, 3]],
                [[1, 0, 1], [2, 0, 1]],
            ],
        )
        self.assertEqual(u.shape, torch.Size([4, 2, 3]))
        res = sut._calc_args_softmax(u, torch.Tensor([1, 2, 3]))
        te.assert_close(
            res,
            torch.Tensor([[6.0, 20.0], [11.0, 13.0], [4.0, 13.0], [4.0, 5.0]]),
        )

    def test_softmax(self):
        sut = a.AttentionModel(1)
        res = sut._sofmax(torch.Tensor([[1, 2], [3, 0], [3, 1]]))
        te.assert_close(
            res[0][0] + res[1][0] + res[2][0], torch.Tensor([1])[0]
        )
        te.assert_close(
            res[0][1] + res[1][1] + res[2][1], torch.Tensor([1])[0]
        )

    def test_embed(self):
        sut = a.AttentionModel(1)
        h = torch.Tensor(
            [
                [[1, 1, 1], [2, 3, 4]],
                [[1, 2, 2], [1, 3, 2]],
                [[1, 0, 1], [2, 1, 3]],
                [[1, 0, 1], [2, 0, 1]],
            ],
        )
        alpha = torch.Tensor([[1, 2], [0, 1], [2, 1], [0, 2]])
        res = sut._embed(h, alpha)
        te.assert_close(res, torch.Tensor([[3, 1, 3], [11, 10, 15]]))
