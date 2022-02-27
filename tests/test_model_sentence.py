import unittest
import torch
import torch.nn as nn
import torch.testing as te
import torch.utils.data as da
import han.vocabulary as v
import tests.marker as marker
import tests.ag_news as ag
import han.model_sentece as m


class HierarchicalAttentionSentenceNetworkTestCase(unittest.TestCase):
    def test(self):
        sut = m.HierarchicalAttentionSentenceNetwork(10, 0)
        u = torch.Tensor(
            [
                [[1, 2, 3, 1], [3, 4, 1, 2], [2, 2, 1, 3]],
                [[5, 6, 2, 2], [7, 3, 2, 1], [3, 2, 1, 4]],
            ]
        )
        self.assertEqual(torch.Size([2, 3, 4]), u.shape)
        context = torch.Tensor([1, 1, 1, 1])
        res = sut.mul_context_vector(u, context)

        te.assert_close(res, torch.Tensor([[7, 10, 8], [15, 13, 10]]))

    def test_word_softmax(self):
        sut = m.HierarchicalAttentionSentenceNetwork(10, 0)
        x = torch.Tensor([[1, 2], [3, 4], [5, 1]])
        res = sut.calc_softmax(x)

        te.assert_close(
            torch.Tensor([res[0][0] + res[1][0] + res[2][0]]),
            torch.Tensor([1]),
        )

    def test_calc_setence_vector(self):
        alpha = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(torch.Size([2, 3]), alpha.shape)
        h = torch.Tensor(
            [
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            ]
        )
        self.assertEqual(torch.Size([2, 3, 4]), h.shape)
        sut = m.HierarchicalAttentionSentenceNetwork(10, 0)
        res = sut.calc_sentence_vector(alpha, h)
        te.assert_close(
            res, torch.Tensor([[5, 5, 5, 5], [7, 7, 7, 7], [9, 9, 9, 9]])
        )

    def test_pack_embeddings(self):
        a = torch.Tensor(
            [
                [[1, 1], [2, 2], [8, 8]],
                [[4, 4], [5, 5], [0, 0]],
                [[0, 0], [7, 7], [0, 0]],
            ]
        )
        self.assertEqual(torch.Size([3, 3, 2]), a.shape)
        sut = m.HierarchicalAttentionSentenceNetwork(10, 0)
        res = sut.pack_embeddings(
            a,
            [2, 3, 1],
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


@unittest.skipUnless(marker.run_integration_tests, marker.skip_reason)
class IntegrationTestCase(unittest.TestCase):
    def test(self):
        """
        References
        ----------
        https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation
        https://developers.google.com/machine-learning/guides/text-classification/step-4
        """
        train: ag.AGNewsDataset = ag.get_train()
        size = len(train)
        vocabulary: v.Vocabulary = ag.build_ag_news_vocabulary(train)
        train_data_loader: da.DataLoader = da.DataLoader(train, batch_size=10)
        model = m.HierarchicalAttentionSentenceNetwork(
            len(vocabulary), vocabulary.pad_index
        )
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for batch_index, batch in enumerate(train_data_loader):
            labels: list[int] = [item[0] for item in batch]
            tokens: list[list[str]] = [item[1] for item in batch]
            word_tensor, lengths = vocabulary.create_matrix(tokens)
            pred = model(word_tensor, lengths)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_index % 100 == 0:
                loss, current = loss.item(), batch_index % len(labels)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    IntegrationTestCase().test()
