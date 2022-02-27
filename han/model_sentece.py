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
        mlp_output_size: int = 100,
    ):
        """Take hyper parameters.

        vocabulary size should count the padding indice.

        References
        ----------
        https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#gru

        """
        super(HierarchicalAttentionSentenceNetwork, self).__init__()
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
        self.mlp = nn.Linear(gru_hidden_size * 2, mlp_output_size)
        self.tanh = nn.Tanh()
        self.context_weights = nn.Parameter(torch.Tensor(mlp_output_size, 1))

    def forward(
        self, x: torch.Tensor, lengths: list[int]
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """Calculate sentence vectors, and attentions.

        x is a word index matrix. The shape is
        (batch size, number of the words in the longest sentence)

        Each item of lengths is the number of the words in each
        sentence before padding.

        """
        # x.shape is (longest length, batch size, embedding dim)
        x: torch.Tensor = self.embedding(x)
        x: r.PackedSequence = self.pack_embeddings(x, lengths)
        h: torch.Tensor = self.gru(x)[0]
        del x
        # MLP cannot accept any packed sequences.
        h, _ = r.pad_packed_sequence(h)
        u: torch.Tensor = self.tanh(self.mlp(h))
        u = self.mul_context_vector(u, self.context_weights)
        alpha = self.calc_softmax(u)
        del u
        return self.calc_sentence_vector(alpha, h), alpha

    def pack_embeddings(
        self, x: torch.Tensor, lengths: list[int]
    ) -> r.PackedSequence:
        """Pack padded and embedded words.

        The shape of x is
        (the longest length of the sentences, batch size, embedding dim).

        """
        return r.pack_padded_sequence(x, lengths, enforce_sorted=False)

    def mul_context_vector(self, u: torch.Tensor, context_param: torch.Tensor):
        """Calulate arguments for softmax.

        The shape of u is
        (the number of words in a sentence, batch size, dimention of word)
        The number of words contains padding.
        The shape of context_param is (dimention of word).
        Return (the number of words in a sentence, batch size).

        """
        return torch.matmul(u, context_param)

    def calc_softmax(self, x: torch.Tensor):
        """Calculate softmax.

        The shape of x is (the number of words in a sentence, word dimension).
        Return the tensor with (the number of words) shape.

        """
        return nn.Softmax(dim=0)(x)

    def calc_sentence_vector(self, alpha: torch.Tensor, h: torch.Tensor):
        """Calculate word or doc vector.

        The shape of alpha is
        (the number of words or senteces, batch size, 1).

        The shape of h is
        (the number of words or sentences, batch size, dimention).

        """
        return torch.sum(torch.mul(alpha.expand_as(h), h), 0)
