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
        embedding: nn.Embedding,
        gru_hidden_size: int,
        sentence_dim: int,
    ):
        """Take hyper parameters.

        `vocabulary_size` should count the padding indice.  Use
        `SentenceModelFactory` instead of calling this constructor.

        """
        super(SentenceModel, self).__init__()
        self._embedding = embedding
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#gru
        self._gru = nn.GRU(
            input_size=self._embedding.embedding_dim,
            hidden_size=gru_hidden_size,
            bidirectional=True,
        )
        self._attention_model = a.AttentionModel(
            gru_hidden_size * 2, sentence_dim
        )
        self.gru_hidden_size = gru_hidden_size
        self.sentence_dim = sentence_dim

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
        x: torch.Tensor = self._embedding(x)
        x: r.PackedSequence = self._pack_embeddings(x, lengths)
        x: torch.Tensor = self._gru(x)[0]
        # Linear cannot accept any packed sequences.
        x, _ = r.pad_packed_sequence(x)
        return self._attention_model(x)

    def sparse_dense_parameters(
        self,
    ) -> t.Tuple[list[nn.parameter.Parameter], list[nn.parameter.Parameter]]:
        """Return the parameters for sparse and dense parameters."""
        if self._embedding.sparse:
            sparse = list(self._embedding.parameters())[0]
            return [sparse], [p for p in self.parameters() if p is not sparse]
        return [], list(self._embedding.parameters())

    def _get_lengths(self, x: list[torch.Tensor]) -> list[int]:
        """Get the lengths of each item."""
        return [e.size()[0] for e in x]

    def _pad_sequence(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Pad a list of variable neght tensors with `self.padding_idx`."""
        return r.pad_sequence(x, padding_value=self._embedding.padding_idx).to(
            torch.int
        )

    def _pack_embeddings(
        self, x: torch.Tensor, lengths: list[int]
    ) -> r.PackedSequence:
        """Pack padded and embedded words.

        The shape of `x` is
        (the longest length of the sentences, batch size, embedding dim).

        """
        return r.pack_padded_sequence(x, lengths, enforce_sorted=False)


class SentenceModelFactory:
    """Create `SentenceModel`."""

    def __init__(self):
        """No parameters."""

    def create(
        self,
        vocabulary_size: int,
        padding_idx: t.Optional[int] = None,
        embedding_dim: t.Optional[int] = None,
        embedding_sparse: t.Optional[bool] = None,
        gru_hidden_size: t.Optional[int] = None,
        sentence_dim: t.Optional[int] = None,
    ) -> SentenceModel:
        """Take hyper parameters."""
        return self._create(
            embedding=nn.Embedding(
                num_embeddings=vocabulary_size,
                embedding_dim=get_default(embedding_dim, 200),
                padding_idx=get_default(padding_idx, 0),
                sparse=get_default(embedding_sparse, True),
            ),
            gru_hidden_size=gru_hidden_size,
            sentence_dim=sentence_dim,
        )

    def use_pretrained(
        self,
        embeddings: torch.Tensor,
        freeze: t.Optional[bool] = None,
        padding_idx: t.Optional[int] = None,
        embedding_sparse: t.Optional[int] = None,
        gru_hidden_size: t.Optional[int] = None,
        sentence_dim: t.Optional[int] = None,
    ) -> SentenceModel:
        """Use a pretrained word embedding."""
        return self._create(
            embedding=nn.Embedding.from_pretrained(
                embeddings,
                get_default(freeze, True),
                padding_idx=get_default(padding_idx, 0),
                sparse=get_default(embedding_sparse, True),
            ),
            gru_hidden_size=gru_hidden_size,
            sentence_dim=sentence_dim,
        )

    def _create(
        self,
        embedding: nn.Embedding,
        gru_hidden_size: t.Optional[int] = None,
        sentence_dim: t.Optional[int] = None,
    ):
        return SentenceModel(
            embedding=embedding,
            gru_hidden_size=get_default(gru_hidden_size, 50),
            sentence_dim=get_default(sentence_dim, 100),
        )


class SentenceClassifier(nn.Module):
    """Use `SentenceModel` for a multi class text classification."""

    def __init__(
        self,
        sentence_model: SentenceModel,
        num_of_classes: int,
    ):
        """`num_of_classes` is the number of the classes.

        It also takes `SentenceModel`. Note that use
        `SentenceClassifierFactory`.

        """
        super(SentenceClassifier, self).__init__()
        self._sentence_model = sentence_model
        self._linear = nn.Linear(
            self._sentence_model.sentence_dim, num_of_classes
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Calculate sentence vectors, and attentions.

        x is a list of sentences.
        A sentence is a tensor that each word index.

        """
        x, alpha = self._sentence_model(x)
        x = self._linear(x)
        if self.training:
            return x, alpha
        else:
            return nn.functional.softmax(x, dim=1), alpha

    def sparse_dense_parameters(
        self,
    ) -> t.Tuple[list[nn.parameter.Parameter], list[nn.parameter.Parameter]]:
        """Return the parameters for sparse and dense parameters.

        The first one for sparse, the second is for dense.

        """
        sparse = self._sentence_model.sparse_dense_parameters()[0]
        return sparse, [
            p for p in self.parameters() if not [s for s in sparse if s is p]
        ]


class SentenceClassifierFactory:
    """Create `SentenceClassifier`."""

    def __init__(self):
        """Take no parameters."""
        self._factory = SentenceModelFactory()

    def create(
        self,
        num_classes: int,
        vocabulary_size: int,
        padding_idx=None,
        embedding_dim=None,
        embedding_sparse=None,
        gru_hidden_size=None,
        sentence_dim=None,
    ) -> SentenceClassifier:
        """Create a classifier."""
        sentence_model: SentenceModel = self._factory.create(
            vocabulary_size=vocabulary_size,
            padding_idx=padding_idx,
            embedding_dim=embedding_dim,
            embedding_sparse=embedding_sparse,
            gru_hidden_size=gru_hidden_size,
            sentence_dim=sentence_dim,
        )
        return self._create(sentence_model, num_classes=num_classes)

    def use_pretrained(
        self,
        num_classes: int,
        embeddings: torch.Tensor,
        freeze: t.Optional[bool] = None,
        padding_idx: t.Optional[int] = None,
        embedding_sparse: t.Optional[int] = None,
        gru_hidden_size: t.Optional[int] = None,
        sentence_dim: t.Optional[int] = None,
    ) -> SentenceClassifier:
        """Use pretrained embeddings."""
        return self._create(
            sentence_model=self._factory.use_pretrained(
                embeddings=embeddings,
                freeze=freeze,
                padding_idx=padding_idx,
                embedding_sparse=embedding_sparse,
                gru_hidden_size=gru_hidden_size,
                sentence_dim=sentence_dim,
            ),
            num_classes=num_classes,
        )

    def _create(self, sentence_model: SentenceModel, num_classes: int):
        return SentenceClassifier(sentence_model, num_classes)


def get_default(v, default):
    """Return `default` if `v` is `None`."""
    return default if v is None else v
