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
        mlp_output_size: int = 100,
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
        self.word_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden_size,
            bidirectional=True,
        )
        self.gru_hidden_size = gru_hidden_size
        self.word_mlp = nn.Linear(gru_hidden_size * 2, mlp_output_size)
        self.word_context_weight = nn.Parameter(
            torch.Tensor(mlp_output_size, 1)
        )

    def forward(self, texts: t.Iterable[str]):
        """Pass dataset to the layers."""
        x: torch.Tensor = self.vocabulary.forward(texts)
        x: torch.Tensor = self.embedding(x)
        # the dimenstion of the below x is
        # (  the max length of a text in texts
        #  , batch size
        #  , 2 * self.gru_hidden_size)
        x, h = self.word_gru(x)
        u: torch.Tensor = self.word_mlp(x)
        u: torch.Tensor = torch.nn.Tanh(u)
        u = self.mul_context_vector(u, self.word_context_weight)
        u = self.calc_softmax(u)
        u = self.calc_word_doc_vector(u, x)
        # (L, N, D * H_{out})(L,N,D∗H ​)

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

    def calc_word_doc_vector(self, alpha: torch.Tensor, h: torch.Tensor):
        """Calculate word or doc vector.

        The shape of alpha is
        (the number of words or senteces, batch size).

        The shape of h is
        (the number of words or sentences, batch size, dimention).

        """
        num_words = alpha.shape[0]
        batch = []
        for item_indice in range(alpha.shape[1]):
            temp = []
            for word_indice in range(num_words):
                temp.append(
                    alpha[word_indice][item_indice]
                    * h[word_indice][item_indice]
                )
            batch.append((torch.sum(torch.vstack(temp), 0)))
        return torch.vstack(batch)
