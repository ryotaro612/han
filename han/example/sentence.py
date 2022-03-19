"""Illustrate usage of `SentenceClassifier`."""
import torch
import torch.nn as nn
import torch.utils.data as da
from ..example import ag_news as ag
from ..encode import sentence as s
from .. import token
from ..model import sentence as se


def train_agnews_classifier(model_path: str, encoder_path: str):
    """Fit a `SentenceClassifier` on AG News."""
    agnews_train: ag.AGNewsDataset = ag.AGNewsDatasetFactory().get_train()
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
    model = se.SentenceClassifier(
        num_of_classes=agnews_train.num_of_classes(),
        vocabulary_size=len(vocabulary) + 1,
    )
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=20, eta_min=0.001
    # )
    epoch = 2
    total_acc = 0
    total_count = 0
    count_at_last_checkpoint = 0
    for _ in range(epoch):
        for sentences, labels in dataloader:
            pred, _ = model(sentences)
            optimizer.zero_grad()
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
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
        # scheduler.step()


if __name__ == "__main__":
    train_agnews_classifier(None, None)
