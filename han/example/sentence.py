"""Illustrate usage of `SentenceClassifier`."""
import torch
import torch.nn as nn
import torch.utils.data as da
import torchmetrics as metrics
from ..example import ag_news as ag
from ..encode import sentence as s
from .. import token
from ..model import sentence as se


def train_agnews_classifier(model_path: str, encoder_path: str):
    """Fit a `SentenceClassifier` on AG News."""
    agnews_train: ag.AGNewsDataset = ag.AGNewsDatasetFactory().get_train(
        limit=2000
    )
    tokenizer = token.Tokenizer()
    vocabulary = ag.build_ag_news_vocabulary(agnews_train, tokenizer)
    encoder = s.SentenceEncoder(vocab=vocabulary, tokenizer=tokenizer)
    # torch.save(encoder, encoder_path)
    dataloader = da.DataLoader(
        agnews_train,
        batch_size=10,
        shuffle=True,
        collate_fn=ag.AgNewsCollateSentenceFn(encoder, False),
    )
    num_classes = agnews_train.num_of_classes()
    model = se.SentenceClassifier(
        num_of_classes=num_classes,
        vocabulary_size=len(vocabulary) + 1,
    )
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    sparse_params, dense_params = model.sparse_dense_parameters()
    dense_optimizer = torch.optim.Adam(dense_params)
    sparse_optimizer = torch.optim.SparseAdam(sparse_params)
    sparse_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        sparse_optimizer, T_max=20, eta_min=0.001
    )
    dense_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dense_optimizer, T_max=20, eta_min=0.001
    )
    epoch = 2
    total_acc = 0
    total_count = 0
    count_at_last_checkpoint = 0
    me = metrics.MetricCollection(
        [
            metrics.Recall(num_classes=num_classes, average="macro"),
            metrics.Precision(num_classes=num_classes, average="macro"),
            metrics.F1Score(num_classes=num_classes, average="macro"),
        ]
    )
    # average_precision = metrics.AveragePrecision(num_classes)
    for _ in range(epoch):
        for sentences, labels in dataloader:
            pred, _ = model(sentences)
            sparse_optimizer.zero_grad()
            dense_optimizer.zero_grad()
            loss = loss_fn(pred, labels)
            loss.backward()
            sparse_optimizer.step()
            dense_optimizer.step()

            me(pred.argmax(1), labels)
            # average_precision(pred, labels)
            total_acc += (pred.argmax(1) == labels).sum().item()
            total_count += len(labels)

            if count_at_last_checkpoint + 100 < total_count:
                count_at_last_checkpoint = total_count
                loss = loss.item()

                acc = total_acc / total_count
                print(
                    f"loss: {loss:>7f} "
                    f"acc: {acc:>7f} "
                    f"{total_count:>5d}"
                )
        print(me.compute())
        # print(average_precision.compute())
        me.reset()
        # average_precision.reset()
        sparse_scheduler.step()
        dense_scheduler.step()


if __name__ == "__main__":
    train_agnews_classifier(None, None)
