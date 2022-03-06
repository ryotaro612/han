"""Place `HierarchicalAttentionNetwork`."""
import typing as t
import torch
import torch.nn as nn
from . import sentence as s


class HierarchicalAttentionNetwork(nn.Module):
    """Document embedding."""

    def __init__(
        self,
        vocabulary_size: int,
        padding_idx: int = 0,
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

        x: list[t.Tuple[int, int, torch.Tensor]] = sorted(
            [
                (document_i, sentence_i, sentence)
                for document_i, document in zip(range(len(x)), x)
                for sentence_i, sentence in zip(range(len(document)), document)
            ],
            key=lambda e: len(e[2]),
            reverse=True,
        )
        order = dict(
            (index, (document_i, sentence_i))
            for index, (document_i, sentence_i, _) in enumerate(x)
        )
        print(order)
        res = self.han([e[2] for e in x])
        print(res)
        raise NotImplementedError()


class HierarchicalAttentionNetworkClassifier(nn.Module):
    """Classification."""

    def __init__(
        self,
        vocabulary_size: int,
        num_of_classes: int,
        padding_idx: int = 0,
        embedding_dim: int = 200,
        gru_hidden_size: int = 50,
        linear_output_dim: int = 100,
    ):
        """Take hyper parameters."""
        self.han = HierarchicalAttentionNetwork(
            vocabulary_size=vocabulary_size,
            padding_idx=padding_idx,
            embedding_dim=embedding_dim,
            gru_hidden_size=gru_hidden_size,
            output_dim=linear_output_dim,
        )
        self.linear = nn.Linear(linear_output_dim, num_of_classes)

    def forward(self, x: list[list[torch.Tensor]]):
        """Embbed text index into document vectors."""
        x, _ = self.han(x)
        return self.linear(x)
