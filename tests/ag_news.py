"""Preprocessed AG NEWS dataset."""
import typing as t
import torchtext.datasets as d
import torch.utils.data as da
import torchtext.data as td
import han.vocabulary as v


class AGNewsDataset(da.Dataset):
    """A wrapper of ag news.

    The datasource is
    https://pytorch.org/text/stable/datasets.html#torchtext.datasets.AG_NEWS
    """

    def __init__(self, raw_dataset: t.Sequence[t.Tuple[int, list[str]]]):
        """Take raw ag news dataset."""
        self.raw_dataset = raw_dataset

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self.raw_dataset)

    def __getitem__(self, i):
        """Get elements."""
        if type(i) == int:
            return self.raw_dataset[i]
        return self.raw_dataset.__getitem__(i)


def get_train(limit: int = 120000) -> AGNewsDataset:
    """Get training dataset.

    limit should be equal or less than 120,000.

    """
    train = d.AG_NEWS()[0]
    tokenizer = td.get_tokenizer("basic_english")
    items = []
    for _ in range(limit):
        label, text = next(train)
        tokens = tokenizer(text)
        items.append((label, tokens))
    return AGNewsDataset(items)


def build_ag_news_vocabulary(agnews: AGNewsDataset) -> v.Vocabulary:
    """Learn vocabulary."""
    return v.build_vocabulary((tokens for label, tokens in agnews))
