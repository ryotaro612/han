"""Preprocessed AG NEWS dataset."""
import typing as t
import torch
import torchtext.datasets as d
import torch.utils.data as da
import torchtext.data as td
import torchtext.vocab as vo
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


class AgNewsCollateSentenceFn:
    """`__call__` emits list of tensors."""

    def __init__(self, vocabulary: vo.Vocab, enforce_sorted: bool):
        """Take learned `vocabulary`."""
        self.vocabulary = vocabulary
        self.enforce_sorted = enforce_sorted

    def __call__(
        self, batch: list[t.Tuple[int, list[str]]]
    ) -> t.Tuple[list[torch.Tensor], torch.Tensor]:
        """Return the list of sentences, and labels.

        Sort the batch by the length of a sentence in a decreasing order,

        The second item is typed `torch.long` because training feature
        requred long typed labels.

        """
        if self.enforce_sorted:
            batch = sorted(batch, key=lambda e: len(e[1]), reverse=True)
        labels: torch.Tensor = torch.Tensor([item[0] for item in batch]).to(
            torch.long
        )
        index_sentences = [
            torch.Tensor(self.vocabulary.forward(item[1])) for item in batch
        ]
        return index_sentences, labels


class AgNewsCollateDocumentFn:
    """`__call__` emits a tuple of documents and labels.

    The Documents are typed `list[list[torch.Tensor]]`.
    """

    def __init__(self, vocabulary: vo.Vocab):
        """Take learned `vocabulary`."""
        self.vocabulary = vocabulary

    def __call__(
        self, batch: list[t.Tuple[int, list[str]]]
    ) -> t.Tuple[list[list[torch.Tensor]], torch.Tensor]:
        """Return indexed documents and labels.

        Return a tuple. The first item is documents. The second item
        is labels.

        """
        labels: torch.Tensor = torch.Tensor([item[0] for item in batch]).to(
            torch.long
        )
        documents = []
        for text in (item[1] for item in batch):
            document = []
            sentence = []
            for word, index in zip(text, self.vocabulary.forward(text)):
                sentence.append(index)
                if word == ".":
                    document.append(torch.Tensor(sentence))
                    sentence = []

            if len(sentence) > 0:
                document.append(torch.Tensor(sentence))
            documents.append(document)
        return documents, labels


class AGNewsDatasetFactory:
    """Load and tokenize AG_NEWS."""

    def get_train(self, limit: int = 120000) -> AGNewsDataset:
        """Get train data."""
        if limit > 120000:
            raise RuntimeError(
                f"{limit} is greater than the number of the total train data."
            )
        train = d.AG_NEWS(split="train")
        return self._tokenize(train, limit)

    def get_test(self, limit: int = 7600) -> AGNewsDataset:
        """Get train data."""
        if limit > 7600:
            raise RuntimeError(
                f"{limit} is greater than the number of the total test data."
            )
        test = d.AG_NEWS(split="test")
        return self._tokenize(test, limit)

    def _tokenize(
        self, stream: t.Iterator[t.Tuple[int, str]], limit: int
    ) -> AGNewsDataset:
        tokenizer = td.get_tokenizer("basic_english")
        items = []
        for _ in range(limit):
            label, text = next(stream)
            items.append((label - 1, tokenizer(text)))
        return AGNewsDataset(items)


def build_ag_news_vocabulary(agnews: AGNewsDataset) -> vo.Vocab:
    """Learn the vocabulary of `agnews`."""
    return v.build_vocabulary((tokens for _, tokens in agnews))
