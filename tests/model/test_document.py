import unittest
import torch
import torch.nn as nn
import torch.utils.data as da
import han.model.document as d
import tests.marker as marker
import han.example.ag_news as ag


class DocumentModelTestCase(unittest.TestCase):
    def test_forward(self):

        sut = d.DocumentModel(10)
        x, sentence_alpha, word_alpha, doc_lens = sut(
            [
                [torch.Tensor([2])],
                [torch.Tensor([1, 2, 3]), torch.Tensor([2, 1])],
                [
                    torch.Tensor([1, 2, 3]),
                    torch.Tensor([1]),
                    torch.Tensor([2, 1]),
                ],
                [
                    torch.Tensor([1, 2]),
                    torch.Tensor([1, 2, 4]),
                    torch.Tensor([3]),
                ],
            ]
        )
        self.assertEqual(x.shape, torch.Size([4, sut.doc_dim]))
        self.assertEqual(sentence_alpha.shape, torch.Size([3, 4]))
        self.assertEqual(word_alpha.shape, torch.Size([3, 9]))
        self.assertEqual(doc_lens, [1, 2, 3, 3])
