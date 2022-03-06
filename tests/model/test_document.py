import unittest
import torch.utils.data as da
import han.model.document as d
import tests.marker as marker
import tests.ag_news as ag


class HierachicalAttentionNetworkTestCase(unittest.TestCase):
    def test_smoke(self):
        agnews = ag.AGNewsDataset(
            [
                (
                    0,
                    [
                        "blue",
                        ".",
                    ],
                ),
                (1, ["quick", "brown", "fox", ".", "jumps", "over"]),
                (
                    2,
                    [
                        "fish",
                        "tasty",
                        "brown",
                        "fish",
                        ".",
                        "coke",
                        "tasty",
                        "coffee",
                        "coffee",
                        "cake",
                        ".",
                    ],
                ),
            ]
        )
        vocabulary = ag.build_ag_news_vocabulary(agnews)

        dataloader = da.DataLoader(
            agnews,
            batch_size=3,
            collate_fn=ag.AgNewsCollateDocumentFn(vocabulary),
        )
        model = d.HierarchicalAttentionNetwork(len(vocabulary) + 1)
        for documents, _ in dataloader:
            model(documents)


@unittest.skipUnless(marker.run_integration_tests, marker.skip_reason)
class HierarchicalAttentionNetworkClassifierIntegrationTestCase(
    unittest.TestCase
):
    def test(self):
        agnews_train = ag.AGNewsDatasetFactory().get_train()
        vocabulary = ag.build_ag_news_vocabulary(agnews_train)
        dataloader = da.DataLoader(
            agnews_train,
            batch_size=10,
            collate_fn=ag.AgNewsCollateDocumentFn(vocabulary),
        )
        model = d.HierarchicalAttentionNetworkClassifier(
            len(vocabulary) + 1, 4
        )
        model.train()
        epoch = 2
        for _ in range(epoch):
            for batch_index, () in enumerate(dataloader):
                pass


if __name__ == "__main__":
    HierarchicalAttentionNetworkClassifierIntegrationTestCase().test()
