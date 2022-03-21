"""Preprocessed AG NEWS dataset."""
import typing as t
import torch
import torchtext.datasets as data
import torchtext.data as tdata
import torch.utils.data as da
import torchtext.vocab as vo
from .. import vocabulary as v
from ..encode import sentence as s
from ..encode import document as d


class AGNewsDataset(da.Dataset):
    """A wrapper of ag news.

    Each item is a tuple. The first item is a label.
    The second item is text.

    The source is
    https://pytorch.org/text/stable/datasets.html#torchtext.datasets.AG_NEWS

    """

    def __init__(self, raw_dataset: t.Sequence[t.Tuple[int, str]]):
        """Take labels and tokens of AG news.

        Each item is a tuple.
        The first item is a label of the second one.

        """
        self.raw_dataset = raw_dataset

    def num_of_classes(self) -> int:
        """Return the number of labels."""
        return len(set([i for i, _ in self.raw_dataset]))

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

    def __init__(
        self,
        encoder: s.SentenceEncodeProtocol,
        enforce_sorted: bool,
        device=torch.device("cpu"),
    ):
        """Take an encoder."""
        self._encoder = encoder
        self._enforce_sorted = enforce_sorted
        self._device = device

    def __call__(
        self, batch: list[t.Tuple[int, str]]
    ) -> t.Tuple[list[torch.Tensor], torch.Tensor]:
        """Return the list of sentences, and labels.

        Sort the batch by the length of a sentence in a decreasing order
        if `self.enforce_sorted` is `True`.

        """
        if self._enforce_sorted:
            batch = sorted(batch, key=lambda e: len(e[1]), reverse=True)

        texts = [
            _allocate(text, self._device)
            for text in self._encoder.forward([text for _, text in batch])
        ]
        labels = _allocate(
            torch.Tensor([label - 1 for label, _ in batch]).to(torch.long),
            self._device,
        )
        return texts, labels

    def to(self, device: torch.device) -> "AgNewsCollateSentenceFn":
        """Make `__call__` put retured tensors on `device`."""
        self._device = device
        return self


class AgNewsCollateDocumentFn:
    """`__call__` emits a tuple of documents and labels.

    Represent documents as `list[list[torch.Tensor]]`.

    """

    def __init__(
        self,
        encoder: d.DocumentEncodeProtocol,
        device: torch.device = torch.device("cpu"),
    ):
        """Take an encoder."""
        self._encoder = encoder
        self._device = device

    def __call__(
        self, batch: list[t.Tuple[int, list[str]]]
    ) -> t.Tuple[list[list[torch.Tensor]], torch.Tensor]:
        """Return indexed documents and labels.

        Return a tuple. The first item is documents. The second item
        is labels.

        """
        labels: torch.Tensor = _allocate(
            torch.Tensor([item[0] - 1 for item in batch]).to(torch.long),
            self._device,
        )
        encoded: list[list[torch.Tensor]] = self._encoder.forward(
            [text for _, text in batch]
        )
        encoded = [
            [_allocate(sentence, self._device) for sentence in document]
            for document in encoded
        ]
        return encoded, labels

    def to(self, device: torch.device) -> "AgNewsCollateDocumentFn":
        """Put tensors on `device`, then return them."""
        self._device = device
        return self


class AGNewsDatasetFactory:
    """Load and tokenize AG_NEWS."""

    def get_train(self, limit: t.Optional[int] = None) -> AGNewsDataset:
        """Get train data."""
        if limit is None:
            limit = 120000
        if limit > 120000:
            raise RuntimeError(
                f"{limit} is greater than the number of the total train data."
            )
        train = list(data.AG_NEWS(split="train"))
        return AGNewsDataset(train[:limit])

    def get_test(self, limit: t.Optional[int] = None) -> AGNewsDataset:
        """Get train data."""
        if limit is None:
            limit = 7600
        if limit > 7600:
            raise RuntimeError(
                f"{limit} is greater than the number of the total test data."
            )
        test = list(data.AG_NEWS(split="test"))
        return AGNewsDataset(test[:limit])


def build_ag_news_vocabulary(
    agnews: AGNewsDataset,
    tokenizer: t.Callable[[str], list[str]] = tdata.get_tokenizer(
        "basic_english"
    ),
) -> vo.Vocab:
    """Learn the vocabulary of `agnews`."""
    return v.build_vocabulary((tokenizer(doc) for _, doc in agnews))


def _allocate(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    else:
        return tensor.to(device)
