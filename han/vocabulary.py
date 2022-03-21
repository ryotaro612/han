"""Word embedding."""
import typing as t
import torchtext.vocab as v
import torch


def build_vocabulary(
    sentences: t.Iterator[t.Iterator[str]],
    pad_symbol: str = "<pad>",
    unknown_symbol: str = "<unk>",
) -> v.Vocab:
    """Build vocabulary.

    Each element of `sentences` is a list of words.  The vocabulary
    encode unknown word to the indice of `unknown_symbol`.

    """
    vocab: v.Vocab = v.build_vocab_from_iterator(
        (sentence for sentence in sentences),
        special_first=True,
        specials=[pad_symbol, unknown_symbol],
    )
    vocab.set_default_index(1)
    return vocab


class EmbeddingProtocol(t.Protocol):
    """Provide the format to provide trained embedding.

    The methods of this protocol follows `torchtext.vocab.Vectors` to
    use it.

    """

    @property
    def itos(self) -> list[str]:
        """Correspond to `stoi`."""

    @property
    def vectors(self) -> torch.Tensor:
        """Return embeddings.

        The shape of the tensor is (`len(itos)`, embedding_dim).

        """


class VocabularyProtocol(t.Protocol):
    """Map strings to index."""

    def forward(self, words: list[str]) -> list[int]:
        """Take words and return their index."""

    def __getitem__(self, s: str) -> int:
        """Take a string and return its indice."""

    def __call__(self, words: list[str]) -> list[int]:
        """See `forward`."""

    def __len__(self) -> int:
        """Return the size of the vocabulary."""


class _VocabularyImpl:
    def __init__(self, dictionary: dict[str, int], default_idx: int = 1):
        self._dictionary = dictionary
        self._default_idx = default_idx

    def forward(self, words: list[str]) -> list[int]:
        return [self.__getitem__(word) for word in words]

    def __getitem__(self, s: str) -> int:
        return self._dictionary.get(s, self._default_idx)

    def __call__(self, words: list[str]) -> list[int]:
        return self.forward(words)

    def __len__(self) -> int:
        return len(self._dictionary)


def create_vocab(
    embedding: EmbeddingProtocol,
    pad_symbol: str = "<pad>",
    unknown_symbol: str = "<unk>",
) -> t.Tuple[VocabularyProtocol, torch.Tensor]:
    """Create a tensor that contains pad and unkown symbols.

    Bind `pad_symbol` to 0 and `unknown_symbol` to 1.

    """
    d = dict()
    d[pad_symbol] = 0
    d[unknown_symbol] = 1
    c = 2
    dim = embedding.vectors.shape[1]
    weights = [torch.Tensor([0] * dim), torch.Tensor([0] * dim)]

    for index, word in enumerate(embedding.itos):
        if word not in set([pad_symbol, unknown_symbol]):
            d[word] = c
            c += 1
            weights.append(embedding.vectors[index, :])

    return _VocabularyImpl(d, 1), torch.vstack(weights)
