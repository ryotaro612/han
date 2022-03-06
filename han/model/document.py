"""Place `HierarchicalAttentionNetwork`."""
import torch.nn as nn
from . import sentence as s


class HierarchicalAttentionNetwork(nn.Module):
    """Document embedding."""

    def __init__(
        self,
        vocabulary_size: int,
        padding_idx: int,
        embedding_dim: int = 200,
        gru_hidden_size: int = 50,
        output_dim: int = 100,
    ):
        """Take hyper parameters."""
        super(HierarchicalAttentionNetwork, self).__init__()
        self.han = s.HierarchicalAttentionSentenceNetwork(
            vocabulary_size,
            padding_idx,
            embedding_dim,
            gru_hidden_size,
            output_dim,
        )
        self.gru = nn.GRU(
            input_size=output_dim,
            hidden_size=gru_hidden_size,
            bidirectional=True,
        )

    def forward(self, x: list[list[torch.Tensor]]) -> torch.Tensor:
        """"""
