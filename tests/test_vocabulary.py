import unittest
import torch
import torch.testing as te
import han.vocabulary as v


class VocabularyTestCase(unittest.TestCase):
    def test(self):
        pad_id = 0
        sut = v.build_vocabulary([["a"], ["b", "c"]], pad_id)
        res, lengths = sut.create_matrix([["a", "b"], ["d", "c", "b"]])
        self.assertEquals([2, 3], lengths)
        te.assert_close(
            res,
            torch.Tensor(
                [[sut["a"], pad_id], [sut["b"], sut["c"]], [pad_id, sut["b"]]]
            ),
        )
