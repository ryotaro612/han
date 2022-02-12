"""Define Hierarchical attention network."""
import torch.nn as nn
from . import vocabulary as v


class HierarchicalAttentionNetwork(nn.Module):
    """Define Hierarchical Attention Network."""

    def __init__(self, embedding_dim, num_embeddings=200):
        """Take hyper parameters.

        References
        ----------
        https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#gru

        """
        super(HierarchicalAttentionNetwork, self).__init__()
        self.vocabulary = v.Vocabulary()
        self.embedding = nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            sparse=True,
        )

        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, text):
        """"""
