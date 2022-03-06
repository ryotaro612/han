import unittest
import torch
import torch.testing as tt
import torch.utils.data as da
import han.vocabulary as v
import tests.ag_news as ag


class AgNewsCollateSentenceFnTestCase(unittest.TestCase):
    def test_call(self):

        vocabulary = v.build_vocabulary(
            [["apple", "banana", "cat", "dog", "leopard"]]
        )
        sut = da.DataLoader(
            [
                (1, ["apple"]),
                (3, ["banana", "apple"]),
                (2, ["cat", "dog", "leopard"]),
            ],
            batch_size=3,
            collate_fn=ag.AgNewsCollateSentenceFn(vocabulary),
        )
        for sentence_index, labels in sut:
            self.assertEqual(sentence_index[0].shape, torch.Size([3]))
            self.assertEqual(sentence_index[1].shape, torch.Size([2]))
            self.assertEqual(sentence_index[2].shape, torch.Size([1]))
            tt.assert_close(labels, torch.Tensor([2, 3, 1]).to(torch.long))


class BuildAgNewsVocabularyTestCase(unittest.TestCase):
    def test_learn(self):
        news = ag.AGNewsDatasetFactory().get_train(1)
        vocab = ag.build_ag_news_vocabulary(news)
        self.assertEqual(set(news[0][1]), set(vocab.get_stoi().keys()))
