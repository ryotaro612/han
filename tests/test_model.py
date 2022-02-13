import unittest
import torch
import torch.nn as nn
import torch.testing as te
from han import model as m
import han.vocabulary as v


class TestHierarchicalAttentionNetwork(unittest.TestCase):
    def setUp(self):
        vocab = v.Vocabulary()
        vocab.build(["lorem", "ipsum"])
        self.sut = m.HierarchicalAttentionNetwork(vocab)

    def test(self):

        u = torch.Tensor(
            [
                [[1, 2, 3, 1], [3, 4, 1, 2], [2, 2, 1, 3]],
                [[5, 6, 2, 2], [7, 3, 2, 1], [3, 2, 1, 4]],
            ]
        )
        self.assertEqual(torch.Size([2, 3, 4]), u.shape)
        context = torch.Tensor([1, 1, 1, 1])
        res = self.sut.mul_context_vector(u, context)

        te.assert_close(res, torch.Tensor([[7, 10, 8], [15, 13, 10]]))

    def test_word_softmax(self):
        x = torch.Tensor([[1, 2], [3, 4], [5, 1]])
        res = self.sut.calc_softmax(x)

        te.assert_close(
            torch.Tensor([res[0][0] + res[1][0] + res[2][0]]),
            torch.Tensor([1]),
        )

    def test_calc_embedding(self):
        alpha = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(torch.Size([2, 3]), alpha.shape)
        h = torch.Tensor(
            [
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            ]
        )
        self.assertEqual(torch.Size([2, 3, 4]), h.shape)
        res = self.sut.calc_word_doc_vector(alpha, h)
        te.assert_close(
            res, torch.Tensor([[5, 5, 5, 5], [7, 7, 7, 7], [9, 9, 9, 9]])
        )
        batch = []
        for word_indice in range(alpha.shape[0]):
            print(alpha[word_indice, :])
            print(h[word_indice, :, :])
            a = torch.transpose(
                torch.unsqueeze(alpha[word_indice, :], 0), 1, 0
            ).expand_as(h[word_indice, :, :])
            print(a)
            print(torch.mul(h[word_indice, :, :], a))
            batch.append(
                torch.unsqueeze(torch.mul(h[word_indice, :, :], a), 1)
            )
            print("-")
        print(batch[0].shape)
        print(torch.cat(batch, 1).shape)
        print(torch.sum(torch.cat(batch, 1), 1))
        assert False
