import unittest
import collections
import torch
import torch.testing as te
import torchtext.vocab as vo
import han.vocabulary as v


class VocabularyTestCase(unittest.TestCase):
    def test(self):
        sut = v.build_vocabulary(
            [["blue"], ["blue", "glass"]],
        )
        self.assertEqual(
            set(sut.get_stoi().keys()),
            set(["blue", "glass", "<pad>", "<unk>"]),
        )
        self.assertEqual(sut["a"], 1)


class CreateVocabTestCase(unittest.TestCase):
    def test(self):
        embedding = collections.namedtuple("Temp", "itos")
        embedding.itos = ["android", "cat", "dog"]
        embedding.vectors = torch.vstack(
            [torch.Tensor(5), torch.Tensor(5), torch.Tensor(5)]
        )
        vocab, weights = v.create_vocab(embedding)
        self.assertEqual(vocab["foo"], 1)
        self.assertEqual(vocab["<pad>"], 0)

        te.assert_close(weights[0, :], torch.Tensor([0] * 5))
        te.assert_close(weights[1, :], torch.Tensor([0] * 5))
        te.assert_close(weights[2, :], embedding.vectors[0])
        te.assert_close(weights[3, :], embedding.vectors[1])
        te.assert_close(weights[4, :], embedding.vectors[2])
