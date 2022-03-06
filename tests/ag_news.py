"""Preprocessed AG NEWS dataset."""
import typing as t
import torch
import torchtext.datasets as d
import torch.utils.data as da
import torchtext.data as td
import han.vocabulary as v


class AGNewsDataset(da.Dataset):
    """A wrapper of ag news.

    Each item is a tuple. The first item is a label.
    Each label is between 0 and 3.
    The second one a list of tokens. The type of the tokens are str.

    The source is
    https://pytorch.org/text/stable/datasets.html#torchtext.datasets.AG_NEWS

    """

    def __init__(self, raw_dataset: t.Sequence[t.Tuple[int, list[str]]]):
        """Take labels and tokens of AG news.

        Each item is a tuple.
        The first item is a label of the second one.
        The label is between 0 and 3.

        """
        self.raw_dataset = raw_dataset

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self.raw_dataset)

    def __getitem__(self, i):
        """Get elements."""
        if type(i) == int:
            return self.raw_dataset[i]
        return AGNewsDataset(self.raw_dataset.__getitem__(i))


class AgNewsCollateFn:
    """A class for collate_fn."""

    def __init__(self, vocabulary: v.Vocabulary):
        """Take a learnt vocabulary."""
        self.vocabulary = vocabulary

    def __call__(
        self, batch: list[t.Tuple[int, list[str]]]
    ) -> t.Tuple[torch.Tensor, list[int], torch.Tensor]:
        """Return a triple.

        The first item of a returned values is a tensor of the word
        index matrix. The second one is the lengths of the
        sentences. The third one is the labels.

        """
        labels: torch.Tensor = torch.Tensor([item[0] for item in batch]).to(
            torch.long
        )
        sentences: list[list[str]] = [item[1] for item in batch]
        word_tensor, lengths = self.vocabulary.forward(sentences)
        return word_tensor, lengths, labels


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
        items.append((label - 1, tokens))
    return AGNewsDataset(items)


class ANewsDatasetFactory:
    """Load and tokenize AGNews."""

    def get_train(self, limit: int = 120000) -> AGNewsDataset:
        """Get train data."""
        if limit > 120000:
            raise RuntimeError(
                f"{limit} is greater than the number of the total train data."
            )
        train = d.AG_NEWS(split="train")
        return self._tokenize(train)

    def get_test(self, limit: int = 7600) -> AGNewsDataset:
        """Get train data."""
        if limit > 7600:
            raise RuntimeError(
                f"{limit} is greater than the number of the total test data."
            )
        test = d.AG_NEWS(split="test")
        return self._tokenize(test)

    def _tokenize(
        self, stream: t.Iterator[t.Tuple[int, str]]
    ) -> AGNewsDataset:
        tokenizer = td.get_tokenizer("basic_english")
        return AGNewsDataset(
            [(label - 1, tokenizer(text)) for label, text in stream]
        )


def build_ag_news_vocabulary(
    agnews: AGNewsDataset, pad_index: int = 0
) -> v.Vocabulary:
    """Learn the vocabulary of `agnews`."""
    return v.build_vocabulary(
        ([tokens] for label, tokens in agnews), pad_index=pad_index
    )


def create_dataloader(
    batch_size: t.Optional[int] = 1,
    pad_index: int = 0,
    limit: int = 120000,
) -> t.Tuple[da.DataLoader, int]:
    """Create a DataLoader that provides AGNewDataset.

    This function returns a tuple. The first item is the DataLoader,
    and the second one is the size of the vocabulary.

    The DataLoader emits a tuple for a batch.  They are the tensor of
    a word index, the lengths of the sentences, the labels.

    """
    dataset: da.Dataset = get_train(limit)
    vocabulary: v.Vocabulary = build_ag_news_vocabulary(dataset, pad_index)
    return da.DataLoader(
        dataset, batch_size=batch_size, collate_fn=AgNewsCollateFn(vocabulary)
    ), len(vocabulary)


# def create_train_sentence_dataloader(
#     batch_size: t.Optional[int] = 1,
#     pad_index: int = 0,
#     limit: int = ) -> t.Tuple[da.DataLoader]:
#     """"""
