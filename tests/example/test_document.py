import unittest
import os
import han.example.document as d


class TrainTestCase(unittest.TestCase):
    def setUp(self):
        self.encoder_file = "/tmp/han_document_encoder.pth"
        self.model_file = "/tmp/han_document.pth"

    def test_sparse(self):
        d.train(
            self.encoder_file,
            self.model_file,
            train_num=100,
            test_num=50,
            device="cpu",
        )
        self.assertTrue(os.path.exists(self.encoder_file))
        self.assertTrue(os.path.exists(self.model_file))

    def test_dense(self):
        d.train(
            self.encoder_file,
            self.model_file,
            train_num=100,
            test_num=50,
            embedding_sparse=False,
            device="cpu",
        )
        self.assertTrue(os.path.exists(self.encoder_file))
        self.assertTrue(os.path.exists(self.model_file))

    def tearDown(self):
        for filepath in [self.encoder_file, self.model_file]:
            if os.path.exists(filepath):
                os.remove(filepath)
