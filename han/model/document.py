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
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Take a document index."""
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


class DocumentClassifier(nn.Module):
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
        self.han = DocumentModel(
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
