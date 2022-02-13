import unittest
from . import ag_news as ag


class AGNewsIterableDatasetTestCase(unittest.TestCase):
    def test(self):
        sut = ag.get_train()
        iteration = iter(sut)

        label, text = next(iteration)
        print(label, text)
        self.assertIsInstance(label, int)
        self.assertIsInstance(text, list)
        for word in text:
            self.assertIsInstance(word, str)

        print(len(ag.build_vocabulary()))
        assert False
