import unittest
import torch
import torch.testing as te
import han.model.sentence as m


class SetenceModelTestCase(unittest.TestCase):
    def setUp(self):
        self.sut = m.SentenceModelFactory().create(vocabulary_size=10)

    def test_sparse(self):
        sparse, dense = self.sut.sparse_dense_parameters()
        self.assertEqual(len(sparse), 1)

    def test_pack_embeddings(self):
        a = torch.Tensor(
            [
                [[2, 2], [1, 1], [8, 8]],
                [[5, 5], [4, 4], [0, 0]],
                [[7, 7], [0, 0], [0, 0]],
            ]
        )
        self.assertEqual(torch.Size([3, 3, 2]), a.shape)
        res = self.sut._pack_embeddings(
            a,
            [3, 2, 1],
        )

        te.assert_allclose(
            res.data,
            torch.Tensor(
                [
                    [2.0, 2.0],
                    [1.0, 1.0],
                    [8.0, 8.0],
                    [5.0, 5.0],
                    [4.0, 4.0],
                    [7.0, 7.0],
                ]
            ),
        )

    def test_get_lengths(self):
        res = self.sut._get_lengths(
            [torch.tensor([3, 3, 3]), torch.tensor([2])]
        )
        self.assertEqual(res, [3, 1])

    def test_pad_sequence(self):
        res = self.sut._pad_sequence(
            [torch.tensor([3, 3, 3]), torch.tensor([2])]
        )
        te.assert_close(
            res, torch.tensor([[3, 2], [3, 0], [3, 0]]).to(torch.int)
        )


class SentenceClassifierTestCase(unittest.TestCase):
    def test_softmax(self):
        sut = m.SentenceClassifier(4, 30)

        sut.eval()
        with torch.no_grad():
            res = sut([torch.Tensor([1, 2, 3]), torch.Tensor([1])])[0]
            te.assert_close(
                torch.sum(res[0, :]), torch.tensor(1).to(torch.float)
            )
            te.assert_close(
                torch.sum(res[1, :]), torch.tensor(1).to(torch.float)
            )

    def test_sparse_dense_parameters(self):

        sut = m.SentenceClassifier(4, 30, embedding_sparse=False)
        sparse, dense = sut.sparse_dense_parameters()
        self.assertEqual(sparse, [])
        self.assertEqual(len(dense), len(list(sut.parameters())))
