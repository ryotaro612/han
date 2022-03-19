import unittest
import torch
import torch.testing as tt
import torch.utils.data as da
import han.vocabulary as v
import han.example.ag_news as ag
import torchtext.data as data
import han.encode.sentence as s
import han.encode.document as d


class AgNewsCollateSentenceFnTestCase(unittest.TestCase):
    def test_call(self):

        vocabulary = v.build_vocabulary(
            [["apple", "banana", "cat", "dog", "leopard"]]
        )

        sut = da.DataLoader(
            ag.AGNewsDataset(
                [
                    (1, "apple"),
                    (3, "banana apple"),
                    (2, "cat dog leopard"),
                ]
            ),
            batch_size=3,
            collate_fn=ag.AgNewsCollateSentenceFn(
                s.SentenceEncoder(vocabulary), True
            ),
        )
        for sentence_index, labels in sut:
            self.assertEqual(sentence_index[0].shape, torch.Size([3]))
            self.assertEqual(sentence_index[1].shape, torch.Size([2]))
            self.assertEqual(sentence_index[2].shape, torch.Size([1]))
            tt.assert_close(labels, torch.Tensor([1, 2, 0]).to(torch.int))


class AgNewsCollateDocumentFnTestCase(unittest.TestCase):
    def test(self):
        vocabulary = v.build_vocabulary(
            [
                ["duck", "cat"],
                ["dog", "lion", "penguin", "banana"],
                ["apple"],
            ]
        )
        sut = ag.AgNewsCollateDocumentFn(d.DocumentEncoder(vocab=vocabulary))
        documents, labels = sut(
            [
                (2, "duck lion. cat."),
                (1, "dog"),
                (3, "penguin banana apple."),
            ]
        )
        tt.assert_close(
            documents,
            [
                [
                    torch.Tensor(vocabulary.forward(["duck", "lion", "."])),
                    torch.Tensor(vocabulary.forward(["cat", "."])),
                ],
                [torch.Tensor(vocabulary.forward(["dog"]))],
                [
                    torch.Tensor(
                        vocabulary.forward(["penguin", "banana", "apple", "."])
                    )
                ],
            ],
        )
        tt.assert_close(labels, torch.Tensor([1, 0, 2]).to(torch.int))


class BuildAgNewsVocabularyTestCase(unittest.TestCase):
    def test_learn(self):
        news = ag.AGNewsDatasetFactory().get_train(1)
        tokenizer = data.get_tokenizer("basic_english")
        vocab = ag.build_ag_news_vocabulary(news, lambda x: tokenizer(x))
        self.assertEqual(
            set(tokenizer(news[0][1])), set(vocab.get_stoi().keys())
        )
