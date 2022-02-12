"""Word embedding."""
import typing as t
import torch
import torchtext.vocab as v
from . import token as to


class Vocabulary:
    """Convet texts to an index matrix."""

    def __init__(self):
        """Take no arguments."""
        self.tokenizer: t.Tokenizer = to.Tokenizer()
        self.pad_id = 0

    def build(self, texts: t.Iterator[str]):
        """Build vocaburary."""
        self.vocab: v.Vocab = v.build_vocab_from_iterator(
            (self.tokenizer(text) for text in texts)
        )
        self.vocab.set_default_index(self.pad_id)

    def forward(self, texts: t.Iterator[str]) -> torch.Tensor:
        """Construct the word index matrix."""
        vectors = [
            [self.vocab[word] for word in self.tokenizer(text)]
            for text in texts
        ]
        max_len = 0
        for vector in vectors:
            max_len = max(len(vector), max_len)

        for index in range(len(vectors)):
            vectors[index] += [
                self.pad_id for _ in range(max_len - len(vectors[index]))
            ]
        return torch.Tensor(vectors)

    def __getitem__(self, key: str) -> int:
        """Look up a word."""
        return self.vocab[key]
