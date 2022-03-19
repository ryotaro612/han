"""Model that transforms word index to a sentence vector."""
import typing as t
import torch
import torch.nn as nn
import torch.nn.utils.rnn as r
from . import attention as a


class SentenceModel(nn.Module):
    """Define Hierarchical Attention Network.

    Transform word index to sentence vectors.

    """

    def __init__(
        self,
        vocabulary_size: int,
        padding_idx: t.Optional[int] = None,
        embedding_dim: t.Optional[int] = None,
        gru_hidden_size: t.Optional[int] = None,
        sentence_dim: t.Optional[int] = None,
    ):
        """Take hyper parameters.

        `vocabulary_size` should count the padding indice.

        """
        super(SentenceModel, self).__init__()
        self.padding_idx = get_default(padding_idx, 0)
        embedding_dim = get_default(embedding_dim, 200)
        self.gru_hidden_size = get_default(gru_hidden_size, 50)
        self.sentence_dim = get_default(sentence_dim, 100)
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx,
            sparse=True,
        )
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#gru
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=self.gru_hidden_size,
            bidirectional=True,
        )
        self.attention_model = a.AttentionModel(
            self.gru_hidden_size * 2, self.sentence_dim
        )

    def forward(
        self, x: list[torch.Tensor]
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """Calculate sentence vectors, and attentions.

        `x` is a list of index sentences.  Return a tuple of two
        tensors.  The first one that it transformed x to, and its
        shape is (num of `x`, `self.output_dim`) The second one
        represents attention.  The shape is (the length of the longest
        tensor in `x`, num of `x`).

        """
        lengths = self._get_lengths(x)
        # x.shape is (longest length, batch size)
        x = self._pad_sequence(x)
        # x.shape is (longest length, batch size, embedding dim)
        x: torch.Tensor = self.embedding(x)
        x: r.PackedSequence = self._pack_embeddings(x, lengths)
        x: torch.Tensor = self.gru(x)[0]
        # Linear cannot accept any packed sequences.
        x, _ = r.pad_packed_sequence(x)
        return self.attention_model(x)

    def _get_lengths(self, x: list[torch.Tensor]) -> list[int]:
        """Get the lengths of each item."""
        return [e.size()[0] for e in x]

    def _pad_sequence(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Pad a list of variable neght tensors with `self.padding_idx`."""
        return r.pad_sequence(x, padding_value=self.padding_idx).to(torch.int)

    def _pack_embeddings(
        self, x: torch.Tensor, lengths: list[int]
    ) -> r.PackedSequence:
        """Pack padded and embedded words.

        The shape of `x` is
        (the longest length of the sentences, batch size, embedding dim).

        """
        return r.pack_padded_sequence(x, lengths, enforce_sorted=False)


class SentenceClassifier(nn.Module):
    """Use `SentenceModel` for a multi class text classification."""

    def __init__(
        self,
        num_of_classes: int,
        vocabulary_size: int,
        padding_idx=None,
        embedding_dim=None,
        gru_hidden_size=None,
        sentence_dim=None,
    ):
        """`num_of_classes` is the number of the classes.

        It also takes the parameters that `SentenceModel` accepts.

        """
        super(SentenceClassifier, self).__init__()
        self.han: SentenceModel = SentenceModel(
            vocabulary_size=vocabulary_size,
            padding_idx=padding_idx,
            embedding_dim=embedding_dim,
            gru_hidden_size=gru_hidden_size,
            sentence_dim=sentence_dim,
        )
        self.linear = nn.Linear(self.han.sentence_dim, num_of_classes)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Calculate sentence vectors, and attentions.

        x is a list of sentences.
        A sentence is a tensor that each word index.

        """
        x, alpha = self.han(x)
        return self.linear(x), alpha


def get_default(v, default):
    """Return `default` if `v` is `None`."""
    return default if v is None else v
