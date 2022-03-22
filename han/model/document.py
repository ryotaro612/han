"""Place `HierarchicalAttentionNetwork`."""
import typing as t
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from . import sentence as s
from . import attention as a


class DocumentModel(nn.Module):
    """Document embedding."""

    def __init__(
        self,
        sentence_model: s.SentenceModel,
        doc_gru_hidden_size: int,
        doc_dim: int,
    ):
        """Use `DocumentModelFactory` instead of calling this."""
        super(DocumentModel, self).__init__()
        self._sentence_model = sentence_model
        self._gru = nn.GRU(
            input_size=self._sentence_model.sentence_dim,
            hidden_size=doc_gru_hidden_size,
            bidirectional=True,
        )
        self._attention_model = a.AttentionModel(
            doc_gru_hidden_size * 2, output_dim=doc_dim
        )
        self.doc_dim = doc_dim

    def forward(
        self, x: list[list[torch.Tensor]]
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Take a document index.

        Return a quadruple. The first item is the document embeddings
        of `x`. The shape is (num of docs, `self.doc_dim`). The second
        item is embeddings of the passed documents. The shape is (The
        max number of sentences in a document, num of documents). The
        third item is embedding of the sentences. The shape is (the
        max number of words in a sentence, the num of sentences). The
        fourth is a list of the number of sentences in each document.

        """
        sentences = [sentence for document in x for sentence in document]
        doc_lens = [len(doc) for doc in x]

        x, word_alpha = self._sentence_model(sentences)
        # x is a tuple of tensors.
        x = torch.split(x, doc_lens)
        x = rnn.pad_sequence(x)
        x = rnn.pack_padded_sequence(x, doc_lens, enforce_sorted=False)
        x = self._gru(x)[0]
        # The shape of x is (max num of sentence, num of docs, dim)
        x = rnn.pad_packed_sequence(x)[0]
        x, alpha = self._attention_model(x)
        return x, alpha, word_alpha, doc_lens

    def sparse_dense_parameters(
        self,
    ) -> t.Tuple[list[nn.parameter.Parameter], list[nn.parameter.Parameter]]:
        """Return the parameters for sparse and dense parameters."""
        sparse: list[
            nn.Parameter
        ] = self._sentence_model.sparse_dense_parameters()[0]
        return sparse, [
            p for p in self.parameters() if not [s for s in sparse if s is p]
        ]


class DocumentModelFactory:
    """Provide ways to create `DocumentModel`."""

    def __init__(self):
        """Take no arguments."""
        self._factory = s.SentenceModelFactory()

    def create(
        self,
        vocabulary_size: int,
        padding_idx=None,
        embedding_dim=None,
        embedding_sparse=None,
        sentence_gru_hidden_size=None,
        sentence_dim=None,
        doc_gru_hidden_size: t.Optional[int] = None,
        doc_dim=None,
    ) -> DocumentModel:
        """Create a `DocumentModel`."""
        sentence_model = self._factory.create(
            vocabulary_size=vocabulary_size,
            padding_idx=padding_idx,
            embedding_dim=embedding_dim,
            embedding_sparse=embedding_sparse,
            gru_hidden_size=sentence_gru_hidden_size,
            sentence_dim=sentence_dim,
        )
        return self._create(
            sentence_model=sentence_model,
            doc_gru_hidden_size=doc_gru_hidden_size,
            doc_dim=doc_dim,
        )

    def use_pretrained(
        self,
        embeddings: torch.Tensor,
        freeze: t.Optional[bool] = None,
        padding_idx: t.Optional[int] = None,
        embedding_sparse: t.Optional[int] = None,
        gru_hidden_size: t.Optional[int] = None,
        sentence_dim: t.Optional[int] = None,
        doc_gru_hidden_size: t.Optional[int] = None,
        doc_dim=None,
    ):
        """Use pretraiend embeddings."""
        return self._create(
            sentence_model=self._factory.use_pretrained(
                embeddings=embeddings,
                freeze=freeze,
                padding_idx=padding_idx,
                embedding_sparse=embedding_sparse,
                gru_hidden_size=gru_hidden_size,
                sentence_dim=sentence_dim,
            ),
            doc_gru_hidden_size=doc_gru_hidden_size,
            doc_dim=doc_dim,
        )

    def _create(self, sentence_model, doc_gru_hidden_size, doc_dim):
        return DocumentModel(
            sentence_model,
            doc_gru_hidden_size=s.get_default(
                doc_gru_hidden_size, sentence_model.gru_hidden_size
            ),
            doc_dim=s.get_default(doc_dim, sentence_model.sentence_dim),
        )


class DocumentClassifier(nn.Module):
    """Use `DocumentModel` for document classification."""

    def __init__(
        self,
        document_model: DocumentModel,
        num_classes: int,
    ):
        """Take hyper parameters."""
        super(DocumentClassifier, self).__init__()
        self._document_model = document_model
        self._linear = nn.Linear(self._document_model.doc_dim, num_classes)

    def forward(self, x: list[list[torch.Tensor]]):
        """Embbed text index into document vectors."""
        x, alpha, word_alpha, doc_lens = self._document_model(x)
        x = self._linear(x)
        x = x if self.training else nn.functional.softmax(x, dim=1)
        return x, alpha, word_alpha, doc_lens

    def sparse_dense_parameters(
        self,
    ) -> t.Tuple[list[nn.parameter.Parameter], list[nn.parameter.Parameter]]:
        """Return the parameters for sparse and dense parameters.

        The first one for sparse, the second is for dense.

        """
        sparse = self._document_model.sparse_dense_parameters()[0]
        return sparse, [
            p for p in self.parameters() if not [s for s in sparse if s is p]
        ]


class DocumentClassifierFactory:
    """Create `DocumentClassifier`."""

    def __init__(self):
        """Take no parameters."""
        self._factory = DocumentModelFactory()

    def create(
        self,
        vocabulary_size: int,
        num_classes: int,
        padding_idx=None,
        embedding_dim=None,
        embedding_sparse=None,
        sentence_gru_hidden_size=None,
        sentence_dim=None,
        doc_gru_hidden_size=None,
        doc_dim=None,
    ):
        """Create `DocumentClassifier`."""
        return DocumentClassifier(
            document_model=self._factory.create(
                vocabulary_size=vocabulary_size,
                padding_idx=padding_idx,
                embedding_dim=embedding_dim,
                embedding_sparse=embedding_sparse,
                sentence_gru_hidden_size=sentence_gru_hidden_size,
                sentence_dim=sentence_dim,
                doc_gru_hidden_size=doc_gru_hidden_size,
                doc_dim=doc_dim,
            ),
            num_classes=num_classes,
        )

    def use_pretrained(
        self,
        num_classes: int,
        embeddings: torch.Tensor,
        freeze: t.Optional[bool] = None,
        padding_idx: t.Optional[int] = None,
        embedding_sparse: t.Optional[int] = None,
        gru_hidden_size: t.Optional[int] = None,
        sentence_dim: t.Optional[int] = None,
        doc_gru_hidden_size: t.Optional[int] = None,
        doc_dim=None,
    ):
        """Use pretrained embedding."""
        return DocumentClassifier(
            self._factory.use_pretrained(
                embeddings=embeddings,
                freeze=freeze,
                padding_idx=padding_idx,
                embedding_sparse=embedding_sparse,
                gru_hidden_size=gru_hidden_size,
                sentence_dim=sentence_dim,
                doc_gru_hidden_size=doc_gru_hidden_size,
                doc_dim=doc_dim,
            ),
            num_classes,
        )
