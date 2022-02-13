import unittest
from . import ag_news as ag


class BuildAgNewsVocabularyTestCase(unittest.TestCase):
    def test(self):
        sut = ag.get_train(100)
        res = ag.build_ag_news_vocabulary(sut)
        self.assertGreater(len(res), 0)
