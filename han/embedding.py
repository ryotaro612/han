"""Word embedding."""
import typing as t
import torch
import torchtext.vocab as v
from . import token as to


class Vocabulary:
    """Make texts an index matrix."""

    def __init__(self):
        """Take no arguments."""
        self.tokenizer: t.Tokenizer = to.Tokenizer()
        self.pad: str = "<pad>"

    def build(self, texts: t.Iterator[str]):
        """Build vocaburary."""
        self.vocab: v.Vocab = v.build_vocab_from_iterator(
            (self.tokenizer(text) for text in texts)
        )
        self.vocab.set_default_index(self.get_unknown_id())
        unknown = "<unk>"
        if unknown not in self.vocab:
            self.vocab.insert_token(unknown, self.get_unknown_id())
        self.vocab.append_token(self.pad)

    def forward(self, texts: t.Iterator[str]) -> torch.Tensor:
        """Construct the word index matrix."""
        vectors = [
            [self.vocab[word] for word in self.tokenizer(text)]
            for text in texts
        ]
        max_len = 0
        for vector in vectors:
            max_len = max(len(vector), max_len)

        pad_indice = self.get_pad_id()
        for index in range(len(vectors)):
            vectors[index] = vectors[index] + [
                pad_indice for _ in range(max_len - len(vectors[index]))
            ]
        return torch.Tensor(vectors)

    def get_pad_id(self):
        """Return pad id."""
        return self.vocab[self.pad]

    def get_unknown_id(self):
        """Return unknown id."""
        return 0

    def __getitem__(self, key: str):
        """Look up a word."""
        return self.vocab[key]
