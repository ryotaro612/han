"""Illustrate usage of `SentenceClassifier`."""
import typing as t
import torch.nn as nn
import torchtext.vocab as v
from ..example import ag_news as ag
from ..encode import sentence as s
from ..model import sentence as se
from .. import vocabulary as hv
from . import model as m


def train(
    encoder_path: str,
    model_path: str,
    train_num: t.Optional[int] = None,
    test_num: t.Optional[int] = None,
    embedding_sparse: t.Optional[bool] = None,
    device: t.Optional[str] = None,
    pre_trained: t.Optional[v.Vectors] = None,
):
    """Fit a model on AG News."""
    m.AgNewsTrainer(
        _SentenceTrainImpl(
            embedding_sparse=embedding_sparse, pre_trained=pre_trained
        ),
        device=device,
        train_num=train_num,
        test_num=test_num,
    ).train(encoder_path, model_path)


class _SentenceTrainImpl:
    """Implement `TrainProtocol`."""

    def __init__(
        self,
        embedding_sparse: t.Optional[bool] = None,
        pre_trained: t.Optional[v.Vectors] = None,
    ):
        """Take hyperperameters."""
        self._embedding_sparse = embedding_sparse
        if pre_trained:
            self._vocabulary, self._weights = hv.create_vocab(pre_trained)

    def create_encoder(
        self, vocabulary: v.Vocab, tokenizer: t.Callable[[str], list[str]]
    ) -> s.SentenceEncodeProtocol:
        """Implement the protocol."""
        return s.SentenceEncoder(
            vocab=self._vocabulary
            if hasattr(self, "_vocabulary")
            else vocabulary,
            tokenizer=tokenizer,
        )

    def create_collate_fn(self, encoder) -> ag.AgNewsCollateSentenceFn:
        """Impl the protocol."""
        return ag.AgNewsCollateSentenceFn(encoder, False)

    def create_model(
        self,
        num_classes: int,
        vocabulary_size: int,
    ) -> nn.Module:
        """Impl the protocol."""
        factory = se.SentenceClassifierFactory()
        if hasattr(self, "_weights"):
            return factory.use_pretrained(num_classes, self._weights)
        else:
            return factory.create(
                num_classes=num_classes,
                vocabulary_size=vocabulary_size,
                embedding_sparse=self._embedding_sparse,
            )
