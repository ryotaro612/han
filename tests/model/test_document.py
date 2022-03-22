import unittest
import torch.testing as te
import torch
import han.model.document as d


class DocumentModelTestCase(unittest.TestCase):
    def test_forward(self):
        sut = d.DocumentModelFactory().create(10)
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


class DocumentClassifierTestCase(unittest.TestCase):
    def test_eval(self):
        sut = d.DocumentClassifier(10, 3)
        sut.eval()
        with torch.no_grad():
            res = sut(
                [
                    [torch.Tensor([1, 2]), torch.Tensor([0, 2])],
                    [torch.Tensor([3])],
                ]
            )[0]
        te.assert_close(torch.sum(res[0, :]), torch.tensor(1).to(torch.float))
        te.assert_close(torch.sum(res[1, :]), torch.tensor(1).to(torch.float))
