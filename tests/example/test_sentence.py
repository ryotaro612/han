import unittest
import os

import han.example.sentence as s


class TestTrainCase(unittest.TestCase):
    def setUp(self):
        self.encoder_file = "/tmp/han_sentence_encoder.pth"
        self.model_file = "/tmp/han_sentence.pth"

    def test(self):
        s.train(self.encoder_file, self.model_file, 100, 100)
        self.assertTrue(os.path.exists(self.encoder_file))
        self.assertTrue(os.path.exists(self.model_file))

    def tearDown(self):
        for filepath in [self.encoder_file, self.model_file]:
            if os.path.exists(filepath):
                os.remove(filepath)
