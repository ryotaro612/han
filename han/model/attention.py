"""Attention layers."""
import typing as t
import torch
import torch.nn as nn


class AttentionModel(nn.Module):
    """Attenion layers."""

    def __init__(self, input_dim: int, output_dim: t.Optional[int] = None):
        """Take hyper parameters.

        The default value of `output_dim` is 100.

        """
        if output_dim is None:
            output_dim = 100
        super(AttentionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        seed = nn.init.uniform_(torch.Tensor(output_dim), a=-1.0, b=1.0)
        self.weights = nn.Parameter(seed)

    def forward(self, h: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        """Transform `h`.

        The shape of `h` is (the max number of the words or sentences,
        batch num, dim).

        Return two tensors.  The first one is embedded entities. The
        shape is (the max number of the words or sentences, dim).  The
        second one is attention. The shape is (the max number of the
        words or sentences, batch size).

        """
        u = self.linear(h)
        u = self.tanh(u)
        alpha = self._calc_args_softmax(u, self.weights)
        del u
        h = self._embed(h, alpha)
        return h, alpha

    def _calc_args_softmax(self, u, weights) -> torch.Tensor:
        """Calculate arguments for softmax.

        The shape of u is (the number of items of an entry, batch
        size, dimention of embedded item). The shape of `weights` is
        (dimention of embedded item).  Return (the number of items of
        an entry, batch size).

        """
        return torch.matmul(u, weights)

    def _sofmax(self, x: torch.Tensor) -> torch.Tensor:
        """Pass `x` to a softmax."""
        return nn.functional.softmax(x, dim=0)

    def _embed(self, h: torch.Tensor, alpha: torch.Tensor):
        return torch.sum(
            torch.mul(torch.unsqueeze(alpha, 2).expand_as(h), h), 0
        )
