"""Tokenize text, and encode words to integers."""
import typing
import torch
from .. import vocabulary as v
from .. import token as t


class SentenceEncodeProtocol(typing.Protocol):
    """Encode a text to a tensor."""

    def forward(texts: list[str]) -> list[torch.Tensor]:
        """Transform texts to tensors."""

    def get_vocabulary_size(self) -> int:
        """Return vocabulary size."""


class SentenceEncoder:
    """Implement `SentenceEncodeProtocol`."""

    def __init__(
        self,
        vocab: v.VocabularyProtocol,
        tokenizer: t.Tokenizer = t.Tokenizer(),
    ):
        """`vocab` has vocabulary."""
        self._vocab = vocab
        self.tokenizer = tokenizer

    def forward(self, texts: list[str]) -> list[torch.Tensor]:
        """Implement `forward`."""
        return [
            torch.Tensor(self._vocab(self.tokenizer(text))) for text in texts
        ]

    def get_vocabulary_size(self) -> int:
        """Implement the protocol."""
        return len(self._vocab)
