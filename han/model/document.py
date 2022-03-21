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
        vocabulary_size: int,
        padding_idx=None,
        embedding_dim=None,
        sentence_gru_hidden_size=None,
        sentence_dim=None,
        doc_gru_hidden_size: t.Optional[int] = None,
        doc_dim=None,
    ):
        """Take hyper parameters."""
        super(DocumentModel, self).__init__()
        self.sentence_model = s.SentenceModel(
            vocabulary_size=vocabulary_size,
            padding_idx=padding_idx,
            embedding_dim=embedding_dim,
            gru_hidden_size=sentence_gru_hidden_size,
            sentence_dim=sentence_dim,
        )
        self.doc_dim = s.get_default(doc_dim, self.sentence_model.sentence_dim)
        self.doc_gru_hidden_size = s.get_default(
            doc_gru_hidden_size, self.sentence_model.gru_hidden_size
        )
        self.gru = nn.GRU(
            input_size=self.sentence_model.sentence_dim,
            hidden_size=self.doc_gru_hidden_size,
            bidirectional=True,
        )
        self.attention_model = a.AttentionModel(
            self.doc_gru_hidden_size * 2, output_dim=self.doc_dim
        )

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

        x, word_alpha = self.sentence_model(sentences)
        # x is a tuple of tensors.
        x = torch.split(x, doc_lens)
        x = rnn.pad_sequence(x)
        x = rnn.pack_padded_sequence(x, doc_lens, enforce_sorted=False)
        x = self.gru(x)[0]
        # The shape of x is (max num of sentence, num of docs, dim)
        x = rnn.pad_packed_sequence(x)[0]
        x, alpha = self.attention_model(x)
        return x, alpha, word_alpha, doc_lens

    def sparse_dense_parameters(
        self,
    ) -> t.Tuple[list[nn.parameter.Parameter], list[nn.parameter.Parameter]]:
        """Return the parameters for sparse and dense parameters."""
        sparse: list[
            nn.Parameter
        ] = self.sentence_model.sparse_dense_parameters()[0]

        return sparse, [p for p in self.parameters() if p is not sparse[0]]


class DocumentClassifier(nn.Module):
    """Use `DocumentModel` for document classification."""

    def __init__(
        self,
        vocabulary_size: int,
        num_of_classes: int,
        padding_idx=None,
        embedding_dim=None,
        sentence_gru_hidden_size=None,
        sentence_dim=None,
        doc_gru_hidden_size=None,
        doc_dim=None,
    ):
        """Take hyper parameters."""
        super(DocumentClassifier, self).__init__()
        self.han = DocumentModel(
            vocabulary_size=vocabulary_size,
            padding_idx=padding_idx,
            embedding_dim=embedding_dim,
            sentence_gru_hidden_size=sentence_gru_hidden_size,
            sentence_dim=sentence_dim,
            doc_gru_hidden_size=doc_gru_hidden_size,
            doc_dim=doc_dim,
        )
        self.linear = nn.Linear(self.han.doc_dim, num_of_classes)

    def forward(self, x: list[list[torch.Tensor]]):
        """Embbed text index into document vectors."""
        x, alpha, word_alpha, doc_lens = self.han(x)
        x = self.linear(x)
        x = x if self.training else nn.functional.softmax(x, dim=1)
        return x, alpha, word_alpha, doc_lens
