import unittest
import torch
import torch.nn as nn
import torch.testing as te
import torch.utils.data as da
import torchtext.vocab as vo
import han.model.sentence as m
import tests.marker as marker
import han.example.ag_news as ag


class SetenceModelTestCase(unittest.TestCase):
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


if __name__ == "__main__":
    SentenceClassifierIntegrationTestCase().test()
