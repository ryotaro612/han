import unittest
import torch
import torch.nn as nn
import torch.testing as te
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
        alpha = torch.Tensor(
            [[[1], [2], [3]], [[4], [5], [6]]],
        )
        self.assertEqual(alpha.shape, torch.Size([2, 3, 1]))
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
        pad_index = 0
        dataloader, vocabulary_size = ag.create_dataloader(
            batch_size=10, pad_index=pad_index, limit=120000
        )
        # the size of a batch
        size = len(dataloader)
        mlp_output_size = 100
        # model = m.HierarchicalAttentionSentenceNetworkClassifier(
        #     vocabulary_size,
        #     padding_idx=pad_index,
        #     mlp_output_size=mlp_output_size,
        #     num_of_classes=4,
        # )
        model = m.DebugModel(
            vocabulary_size=vocabulary_size,
            padding_idx=pad_index,
            embedding_dim=200,
            gru_hidden_size=50,
            num_of_classes=4,
        )
        model.train()
        loss_fn = nn.CrossEntropyLoss()
        total_acc = 0
        total_count = 0
        last_process = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for _ in range(2):
            for batch_index, (
                word_index,
                sentence_lengths,
                labels,
            ) in enumerate(dataloader):

                pred = model(word_index, sentence_lengths)
                print(pred.shape)
                loss = loss_fn(pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_acc += (pred.argmax(1) == labels).sum().item()
                total_count += len(labels)
                if last_process + 100 < total_count:
                    last_process = total_count
                    loss = loss.item()
                    acc = total_acc / total_count
                    print(
                        f"loss: {loss:>7f} "
                        f"acc: {acc:>7f} "
                        f"{total_count:>5d}"
                    )


if __name__ == "__main__":
    IntegrationTestCase().test()
