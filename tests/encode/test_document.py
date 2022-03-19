import unittest
import torch
import torch.testing as te
import torchtext.vocab as v
import han.encode.document as d


class DocumentEncoderTestCase(unittest.TestCase):
    def test(self):
        vocab = v.build_vocab_from_iterator([[".", "apple", "is", "tasty"]])
        sut = d.DocumentEncoder(vocab)
        res = sut.forward(["apple is. tasty", "tasty. is apple"])
        te.assert_close(
            res,
            [
                [
                    torch.Tensor([vocab["apple"], vocab["is"], vocab["."]]),
                    torch.Tensor([vocab["tasty"]]),
                ],
                [
                    torch.Tensor([vocab["tasty"], vocab["."]]),
                    torch.Tensor([vocab["is"], vocab["apple"]]),
                ],
            ],
        )
