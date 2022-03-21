import unittest
import os
import han.example.document as d


class TrainTestCase(unittest.TestCase):
    def setUp(self):
        self.encoder_file = "/tmp/han_document_encoder.pth"
        self.model_file = "/tmp/han_document.pth"

    def test(self):
        d.train(self.encoder_file, self.model_file, 300, 100)
        self.assertTrue(os.path.exists(self.encoder_file))
        self.assertTrue(os.path.exists(self.model_file))

    def tearDown(self):
        for filepath in [self.encoder_file, self.model_file]:
            if os.path.exists(filepath):
                os.remove(filepath)
