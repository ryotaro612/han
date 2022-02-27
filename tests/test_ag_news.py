import unittest
import torch
import tests.ag_news as ag


class BuildAgNewsVocabularyTestCase(unittest.TestCase):
    def test_learn(self):
        ag_news = ag.get_train(2)
        sut = ag.build_ag_news_vocabulary(ag_news)
        tensor, lengths = sut.forward(
            [label_text[1] for label_text in ag_news]
        )
        text0_len = len(ag_news[0][1])
        text1_len = len(ag_news[1][1])
        longest_len = max([text0_len, text1_len])
        self.assertEqual(lengths, [text0_len, text1_len])
        self.assertEqual(
            tensor.shape,
            torch.Size([longest_len, 2]),
        )
        print(tensor[:, 0])
        for index, word_index in enumerate(tensor[:, 0]):
            if index < text0_len:
                self.assertGreater(word_index, -1)
            else:
                self.assertEqual(word_index, 0)
        for index, word_index in enumerate(tensor[:, 1]):
            if index < text1_len:
                self.assertGreater(word_index, -1)
            else:
                self.assertEqual(word_index, 0)


# class BuildAgNewsDataLoaderTestCase(unittest.TestCase):
#     def test(self):
#         sut = ag.create_dataloader(10, batch_size=1)
#         for a in sut:
#             print(a[0].shape)
#             print(a)

#         assert False
