"""Illustrate usage of `SentenceClassifier`."""
import typing as t
import os
import logging
import torch
import torch.nn as nn
import torch.utils.data as da
import torchtext.vocab as v
import torchmetrics as tmetrics
from ..example import ag_news as ag
from .. import token

_logger = logging.getLogger(__name__)
_logger.setLevel("INFO")


class TrainProtocol(t.Protocol):
    """Helper for `AgNewsTrainer`."""

    def create_encoder(
        self, vocabulary: v.Vocab, tokenizer: t.Callable[[str], list[str]]
    ):
        """Provide a method that transform text to word index tensors."""

    def create_collate_fn(self, encoder):
        """Create a `collate_fn` that will be passed to `DataLoader`."""

    def create_model(
        self,
        num_classes: int,
        vocabulary_size: int,
    ) -> nn.Module:
        """Create a model."""


class AgNewsTrainer:
    """Implement common steps to train classifiers."""

    def __init__(
        self,
        train_protocol: TrainProtocol,
        device: torch.device = torch.device("cpu"),
        epoch=5,
        train_num=None,
        test_num=None,
    ):
        """Choose a protocol depending on what you want to train."""
        self._train_protocol = train_protocol
        self._device = device
        self._epoch = epoch
        self._train_num = train_num
        self._test_num = test_num

    def train(self, encoder_path: str, model_path: str):
        """Fit a model on AG News."""
        encoder = self._create_encoder(encoder_path)
        torch.save(encoder, encoder_path)

        train_dataloader, test_dataloader = self._load_dataloders(encoder)
        num_classes = 4
        if os.path.exists(model_path):
            _logger.debug(f"Load a model from {model_path}")
            model = torch.load(model_path)
        else:
            model = self._train_protocol.create_model(
                num_classes,
                encoder.get_vocabulary_size(),
            )
        model.to(self._device).train()
        loss_fn = nn.CrossEntropyLoss().to(self._device)

        sparse_params, dense_params = model.sparse_dense_parameters()
        optimizer = _CompositeOptimizer(sparse_params, dense_params)
        scheduler = _CompositeScheduler(optimizer.get_optimizers())

        metrics = tmetrics.MetricCollection(
            [
                tmetrics.Accuracy(num_classes=num_classes),
                tmetrics.Recall(num_classes=num_classes, average="macro"),
                tmetrics.Precision(num_classes=num_classes, average="macro"),
                tmetrics.F1Score(num_classes=num_classes, average="macro"),
            ]
        ).to(self._device)

        for epoch_index in range(self._epoch):
            _logger.info(f"Epoch {epoch_index}")
            model.train()
            metrics.reset()
            for sentences, labels in train_dataloader:
                preds = model(sentences)[0]
                optimizer.zero_grad()
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                metrics(preds.argmax(1), labels)
            scheduler.step()
            self._log("Train", metrics.compute())
            with torch.no_grad():
                model.eval()
                metrics.reset()
                for sentences, labels in test_dataloader:
                    preds = model(sentences)[0]
                    metrics(preds.argmax(1), labels)
                self._log("Test", metrics.compute())
        torch.save(model, model_path)

    def _create_encoder(self, encoder_path):
        if os.path.exists(encoder_path):
            _logger.debug(f"Load an encoder from {encoder_path}")
            return torch.load(encoder_path)
        agnews_train: ag.AGNewsDataset = ag.AGNewsDatasetFactory().get_train()
        tokenizer = token.Tokenizer()
        vocabulary = ag.build_ag_news_vocabulary(agnews_train, tokenizer)
        return self._train_protocol.create_encoder(vocabulary, tokenizer)

    def _load_dataloders(
        self, encoder
    ) -> t.Tuple[da.DataLoader, da.DataLoader]:
        agnews_factory = ag.AGNewsDatasetFactory()
        collate_fn = self._train_protocol.create_collate_fn(encoder).to(
            self._device
        )
        train_dataloader = da.DataLoader(
            agnews_factory.get_train(self._train_num),
            batch_size=50,
            shuffle=True,
            collate_fn=collate_fn,
        )
        test_dataloader = da.DataLoader(
            agnews_factory.get_test(self._test_num),
            batch_size=50,
            collate_fn=collate_fn,
        )
        return train_dataloader, test_dataloader

    def _log(self, title: str, metrics: dict):
        _logger.info(
            f"{title}: "
            f"Accuracy {metrics['Accuracy']:<.3f}, "
            f"Precision {metrics['Precision']:<.3f}, "
            f"Recall {metrics['Recall']:<.3f}, "
            f"F1 {metrics['F1Score']:<.3f}"
        )


class _CompositeOptimizer:
    def __init__(
        self,
        sparse_params: list[torch.nn.Parameter],
        dense_params: list[torch.nn.Parameter],
    ):
        self._optimizers = []
        if sparse_params:
            self._optimizers.append(torch.optim.SparseAdam(sparse_params))

        if dense_params:
            self._optimizers.append(torch.optim.Adam(dense_params))

    def get_optimizers(self) -> list[torch.optim.Optimizer]:
        return self._optimizers

    def zero_grad(self):
        for optimizer in self._optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self._optimizers:
            optimizer.step()


class _CompositeScheduler:
    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self._schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=20, eta_min=0.001
            )
            for optimizer in optimizers
        ]

    def step(self):
        for scheduler in self._schedulers:
            scheduler.step()


def select_device() -> torch.device:
    """Select a device."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        return torch.device("cpu")
