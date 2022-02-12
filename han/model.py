"""Define Hierarchical attention network."""
import torch.nn as nn
import torch
import typing as t
from . import vocabulary as v


class HierarchicalAttentionNetwork(nn.Module):
    """Define Hierarchical Attention Network."""

    def __init__(
        self,
        vocabulary: v.Vocabulary,
        embedding_dim: int = 200,
        gru_hidden_size: int = 50,
    ):
        """Take hyper parameters.

        Note that vocabulary must knows words of the training dataset.

        References
        ----------
        https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#gru

        """
        super(HierarchicalAttentionNetwork, self).__init__()
        self.vocabulary = vocabulary
        self.embedding = nn.Embedding(
            # + 1 is for <pad>.
            num_embeddings=len(self.vocabulary) + 1,
            embedding_dim=embedding_dim,
            padding_idx=self.vocabulary.pad_id,
            sparse=True,
        )
        self.word_encoder_rnn = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=gru_hidden_size,
            bidirectional=True,
        )
        self.gru_hidden_size = gru_hidden_size

    def forward(self, texts: t.Iterable[str]):
        """Pass dataset to the layers."""
        x: torch.Tensor = self.vocabulary.forward(texts)
        x: torch.Tensor = self.embedding(x)
        # the dimenstion of the below matrix is
        # (the max length of the texts, batch size, 2 * self.gru_hidden_size)
        x: torch.Tensor = self.word_encoder_rnn(x)
        x: torch.Tensor = torch.nn.Tanh(x)

        # (L, N, D * H_{out})(L,N,D∗H ​)
