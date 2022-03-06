import unittest
import tests.ag_news as ag


class BuildAgNewsVocabularyTestCase(unittest.TestCase):
    def test_learn(self):
        news = ag.AGNewsDatasetFactory().get_train(1)
        vocab = ag.build_ag_news_vocabulary(news)
        self.assertEqual(set(news[0][1]), set(vocab.get_stoi().keys()))
