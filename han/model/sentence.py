"""Model that transforms word index to a sentence vector."""
import typing as t
import torch
import torch.nn as nn
import torch.nn.utils.rnn as r


class HierarchicalAttentionSentenceNetwork(nn.Module):
    """Define Hierarchical Attention Network.

    Transform word index to a sentence vector.

    """

    def __init__(
        self,
        vocabulary_size: int,
        padding_idx: int,
        embedding_dim: int = 200,
        gru_hidden_size: int = 50,
        output_dim: int = 100,
    ):
        """Take hyper parameters.

        vocabulary size should count the padding indice.

        References
        ----------
        https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#gru

        """
        super(HierarchicalAttentionSentenceNetwork, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            sparse=True,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden_size,
            bidirectional=True,
        )
        self.linear = nn.Linear(gru_hidden_size * 2, output_dim)
        self.tanh = nn.Tanh()
        self.context_weights = nn.Parameter(torch.Tensor(output_dim, 1))

    def forward(
        self, x: list[torch.Tensor]
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """Calculate sentence vectors, and attentions.

        `x` is a list of index sentences.
        `x` should be in the descreasing order of length.

        """
        lengths = self._get_lengths(x)
        # x.shape is (longest length, batch size)
        x = self._pad_sequence(x)
        # x.shape is (longest length, batch size, embedding dim)
        x: torch.Tensor = self.embedding(x)
        x: r.PackedSequence = self._pack_embeddings(x, lengths)
        h: torch.Tensor = self.gru(x)[0]
        del x
        # Linear cannot accept any packed sequences.
        h, _ = r.pad_packed_sequence(h)
        u: torch.Tensor = self.tanh(self.linear(h))
        u = self._mul_context_vector(u, self.context_weights)
        alpha = self._calc_softmax(u)
        del u
        return self._calc_sentence_vector(alpha, h), alpha

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

        The shape of x is
        (the longest length of the sentences, batch size, embedding dim).

        """
        return r.pack_padded_sequence(x, lengths)

    def _mul_context_vector(
        self, u: torch.Tensor, context_param: torch.Tensor
    ):
        """Calulate arguments for softmax.

        The shape of u is
        (the number of words in a sentence, batch size, dimention of word)
        The number of words contains padding.
        The shape of context_param is (dimention of word).
        Return (the number of words in a sentence, batch size).

        """
        return torch.matmul(u, context_param)

    def _calc_softmax(self, x: torch.Tensor):
        """Calculate softmax.

        The shape of x is (the number of words in a sentence, word dimension).
        Return the tensor with (the number of words) shape.

        """
        return nn.Softmax(dim=0)(x)

    def _calc_sentence_vector(self, alpha: torch.Tensor, h: torch.Tensor):
        """Calculate word or doc vector.

        The shape of alpha is
        (the number of words or senteces, batch size, 1).

        The shape of h is
        (the number of words or sentences, batch size, dimention).

        """
        return torch.sum(torch.mul(alpha.expand_as(h), h), 0)


class HierarchicalAttentionSentenceNetworkClassifier(nn.Module):
    """Use `HierachicalAttentionSentenceNetwork' for a multi class problem."""

    def __init__(
        self,
        vocabulary_size: int,
        padding_idx: int,
        linear_output_size: int,
        num_of_classes: int,
    ):
        """`num_of_classes' is the number of the classes."""
        super(HierarchicalAttentionSentenceNetworkClassifier, self).__init__()
        self.han: HierarchicalAttentionSentenceNetwork = (
            HierarchicalAttentionSentenceNetwork(
                vocabulary_size,
                padding_idx=padding_idx,
                output_dim=linear_output_size,
            )
        )
        self.linear = nn.Linear(linear_output_size, num_of_classes)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Calculate sentence vectors, and attentions.

        x is a list of sentences.
        A sentence is a tensor that each word index.

        """
        x, _ = self.han(x)
        return self.linear(x)
