"""Illustrate usage of `SentenceClassifier`."""
import typing as t
import torch
import torch.nn as nn
import torchtext.vocab as v
from ..example import ag_news as ag
from ..encode import sentence as s
from ..model import sentence as se
from . import model as m


def train(encoder_path: str, model_path: str):
    """Fit a model on AG News."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    m.AgNewsTrainer(_SentenceTrainImpl(), device).train(
        encoder_path, model_path
    )


class _SentenceTrainImpl:
    """Implement `TrainProtocol`."""

    def create_encoder(
        self, vocabulary: v.Vocab, tokenizer: t.Callable[[str], list[str]]
    ) -> s.SentenceEncodeProtocol:
        """Implement the protocol."""
        return s.SentenceEncoder(vocab=vocabulary, tokenizer=tokenizer)

    def create_collate_fn(self, encoder) -> ag.AgNewsCollateSentenceFn:
        """Impl the protocol."""
        return ag.AgNewsCollateSentenceFn(encoder, False)

    def create_model(
        self, num_classes: int, vocabulary_size: int
    ) -> nn.Module:
        """Impl the protocol."""
        return se.SentenceClassifier(
            num_of_classes=num_classes,
            vocabulary_size=vocabulary_size,
        )
