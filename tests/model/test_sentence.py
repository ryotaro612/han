import unittest
import torch
import torch.nn as nn
import torch.testing as te
import torch.utils.data as da
import torchtext.vocab as vo
import han.model.sentence as m
import tests.marker as marker
import tests.ag_news as ag


class SetenceModelTestCase(unittest.TestCase):
    def test(self):
        sut = m.SentenceModel(10, 0)
        u = torch.Tensor(
            [
                [[1, 2, 3, 1], [3, 4, 1, 2], [2, 2, 1, 3]],
                [[5, 6, 2, 2], [7, 3, 2, 1], [3, 2, 1, 4]],
            ]
        )
        self.assertEqual(torch.Size([2, 3, 4]), u.shape)
        context = torch.Tensor([1, 1, 1, 1])
        res = sut._mul_context_vector(u, context)

        te.assert_close(res, torch.Tensor([[7, 10, 8], [15, 13, 10]]))

    def test_word_softmax(self):
        sut = m.SentenceModel(10, 0)
        x = torch.Tensor([[1, 2], [3, 4], [5, 1]])
        res = sut._calc_softmax(x)

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
        sut = m.SentenceModel(10, 0)
        res = sut._calc_sentence_vector(alpha, h)
        te.assert_close(
            res, torch.Tensor([[5, 5, 5, 5], [7, 7, 7, 7], [9, 9, 9, 9]])
        )

    def test_pack_embeddings(self):
        a = torch.Tensor(
            [
                [[2, 2], [1, 1], [8, 8]],
                [[5, 5], [4, 4], [0, 0]],
                [[7, 7], [0, 0], [0, 0]],
            ]
        )
        self.assertEqual(torch.Size([3, 3, 2]), a.shape)
        sut = m.SentenceModel(10, 0)
        res = sut._pack_embeddings(
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
        sut = m.SentenceModel(10, 0)
        res = sut._get_lengths([torch.tensor([3, 3, 3]), torch.tensor([2])])
        self.assertEqual(res, [3, 1])

    def test_pad_sequence(self):
        sut = m.SentenceModel(10, 0)
        res = sut._pad_sequence([torch.tensor([3, 3, 3]), torch.tensor([2])])
        te.assert_close(
            res, torch.tensor([[3, 2], [3, 0], [3, 0]]).to(torch.int)
        )


@unittest.skipUnless(marker.run_integration_tests, marker.skip_reason)
class SentenceClassifierIntegrationTestCase(unittest.TestCase):
    def test(self):
        """Dry run.

        References
        ----------
        https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation
        https://developers.google.com/machine-learning/guides/text-classification/step-4

        loss: 0.723289 acc: 0.763470 11990
        loss: 0.412001 acc: 0.763967 12100
        loss: 0.617525 acc: 0.764455 12210

        """
        agnews_train: ag.AGNewsDataset = ag.AGNewsDatasetFactory().get_train()
        vocabulary: vo.Vocab = ag.build_ag_news_vocabulary(agnews_train)
        dataloader = da.DataLoader(
            agnews_train,
            batch_size=10,
            shuffle=True,
            collate_fn=ag.AgNewsCollateSentenceFn(vocabulary, False),
        )
        model = m.SentenceClassifier(
            vocabulary_size=len(vocabulary) + 1,  # 1 is for unknown word.
            num_of_classes=4,
        )
        model.train()
        loss_fn = nn.CrossEntropyLoss()
        total_acc = 0
        total_count = 0
        last_process = 0
        optimizer = torch.optim.Adagrad(model.parameters())
        epoch = 2
        for _ in range(epoch):
            for batch_index, (sentences_index, labels) in enumerate(
                dataloader
            ):

                pred, _ = model(sentences_index)
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


class SortDescreaseTestCase(unittest.TestCase):
    def test(self):
        texts = [
            torch.Tensor([1, 2]),
            torch.Tensor([2]),
            torch.Tensor([3, 2, 1]),
        ]
        res, order = m.sort_descrease(texts)

        te.assert_close(
            res,
            [
                torch.Tensor([3, 2, 1]),
                torch.Tensor([1, 2]),
                torch.Tensor([2]),
            ],
        )
        te.assert_close(order, torch.Tensor([1, 2, 0]).to(torch.int))

    def test_text_is2d(self):
        texts = [
            torch.Tensor([[1, 2, 3]]),
            torch.Tensor([[1, 2], [2, 3]]),
            torch.Tensor([[2], [2], [1]]),
        ]
        res, order = m.sort_descrease(texts)
        te.assert_close(
            res,
            [
                torch.Tensor([[2], [2], [1]]),
                torch.Tensor([[1, 2], [2, 3]]),
                torch.Tensor([[1, 2, 3]]),
            ],
        )
        te.assert_close(order, torch.Tensor([2, 1, 0]).to(torch.int))


if __name__ == "__main__":
    SentenceClassifierIntegrationTestCase().test()
