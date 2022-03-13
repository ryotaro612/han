"""Place `HierarchicalAttentionNetwork`."""
import typing as t
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from . import sentence as s


class DocumentModel(nn.Module):
    """Document embedding."""

    def __init__(
        self,
        vocabulary_size: int,
        padding_idx=None,
        embedding_dim=None,
        gru_hidden_size=None,
        output_dim=None,
        doc_gru_hidden_size: t.Optional[int] = None,
    ):
        """Take hyper parameters."""
        super(DocumentModel, self).__init__()
        self.sentence_model = s.SentenceModel(
            **dict(
                [
                    (k, v)
                    for k, v in [
                        ("vocabulary_size", vocabulary_size),
                        ("padding_idx", padding_idx),
                        ("embedding_dim", embedding_dim),
                        ("gru_hidden_size", gru_hidden_size),
                        ("output_dim", output_dim),
                        ("pre_sorted", False),
                    ]
                    if v is not None
                ]
            )
        )
        self.doc_gru_hidden_size = (
            doc_gru_hidden_size
            if doc_gru_hidden_size
            else self.sentence_model.gru_hidden_size
        )

        self.gru = nn.GRU(
            input_size=self.sentence_model.output_dim,
            hidden_size=self.doc_gru_hidden_size,
            bidirectional=True,
        )

    def forward(self, x: list[list[torch.Tensor]]) -> torch.Tensor:
        """Take a document index."""
        sentences = [sentence for document in x for sentence in document]
        doc_lens = [len(doc) for doc in x]

        x, word_alpha = self.sentence_model(sentences)
        # x is a list of tensors.
        x = torch.split(x, doc_lens)
        print([len(e) for e in x])
        x = rnn.pad_sequence(x)
        x = rnn.pack_padded_sequence(x, doc_lens)
        x = self.gru(x)[0]
        print(x)


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
