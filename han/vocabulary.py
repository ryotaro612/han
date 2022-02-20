"""Word embedding."""
import typing as t
import torch
import torchtext.vocab as v
import torch.nn.utils.rnn as r


class Vocabulary:
    """Convet texts to an index matrix.

    `pad_index` is the padding index.

    """

    def __init__(self, vocab: v.Vocab, pad_index: int = 0):
        """Take a learned vocabulary."""
        self.vocab = vocab
        self.pad_index = pad_index

    def create_matrix(
        self, sentences: t.Iterator[t.Iterator[str]]
    ) -> t.Tuple[torch.Tensor, list[int]]:
        """Construct the word index matrix.

        Return the matrix with (L, B) shape.
        L is the the length of the longest sentece.
        B is the batch size, and same as len(sentences).

        """
        lengths: list[int] = [len(sentence) for sentence in sentences]
        return (
            r.pad_sequence(
                [
                    torch.Tensor([self.vocab[word] for word in words])
                    for words in sentences
                ],
                batch_first=False,
                padding_value=self.pad_index,
            ),
            lengths,
        )

    def __getitem__(self, key: str) -> int:
        """Look up a word."""
        return self.vocab[key]

    def __len__(self) -> int:
        """Return the number of the words of the trained vocabulary."""
        return len(self.vocab)


def build_vocabulary(
    sentences: t.Iterator[t.Iterator[str]], pad_index: int = 0
) -> Vocabulary:
    """Build a vocabulary."""
    vocab: v.Vocab = v.build_vocab_from_iterator(
        (word for words in sentences for word in words)
    )
    vocab.set_default_index(pad_index)
    return Vocabulary(vocab, pad_index)
