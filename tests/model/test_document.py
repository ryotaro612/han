import unittest
import torch
import torch.nn as nn
import torch.utils.data as da
import han.model.document as d
import tests.marker as marker
import tests.ag_news as ag


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


@unittest.skipUnless(marker.run_integration_tests, marker.skip_reason)
class DocumentClassifierIntegrationTestCase(unittest.TestCase):
    def test(self):
        agnews_train = ag.AGNewsDatasetFactory().get_train()
        vocabulary = ag.build_ag_news_vocabulary(agnews_train)
        dataloader = da.DataLoader(
            agnews_train,
            batch_size=10,
            shuffle=True,
            collate_fn=ag.AgNewsCollateDocumentFn(vocabulary),
        )
        model = d.DocumentClassifier(len(vocabulary) + 1, 4)
        model.train()
        loss_fn = nn.CrossEntropyLoss()
        total_acc = 0
        total_count = 0
        last_process = 0
        optimizer = torch.optim.Adagrad(model.parameters())
        epoch = 2
        for _ in range(epoch):
            for batch_index, (doc_index, labels) in enumerate(dataloader):
                pred = model(doc_index)[0]
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
    DocumentClassifierIntegrationTestCase().test()
