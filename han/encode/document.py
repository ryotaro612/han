"""Tokenize text, and encode words to integers."""
import typing
import torch
from .. import vocabulary as v
from .. import token as t


class DocumentEncodeProtocol(typing.Protocol):
    """Encode a text to a tensor."""

    def forward(texts: list[str]) -> list[list[torch.Tensor]]:
        """Transform texts to tensors."""

    def get_vocabulary_size(self) -> int:
        """Return vocabulary size."""


class DocumentEncoder:
    """Implement `DocumentEncodeProtocol`."""

    def __init__(self, vocab: v.VocabularyProtocol):
        """`vocab` has vocabulary."""
        self._vocab = vocab
        self._tokenizer = t.Tokenizer()

    def forward(self, texts: list[str]) -> list[torch.Tensor]:
        """Implement `forward`."""
        documents = []
        for text in texts:
            document = []
            sentence = []
            tokens = self._tokenizer(text)
            for word, index in zip(tokens, self._vocab.forward(tokens)):
                sentence.append(index)
                if word == ".":
                    document.append(torch.Tensor(sentence))
                    sentence = []

            if len(sentence) > 0:
                document.append(torch.Tensor(sentence))
            documents.append(document)
        return documents

    def get_vocabulary_size(self) -> int:
        """Implement the protocol."""
        return len(self._vocab)
