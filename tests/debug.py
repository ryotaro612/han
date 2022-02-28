"""Debug."""
import torch
from torch.cuda import is_available
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchmetrics import Accuracy, F1
from torch import save, stack, no_grad
from torch.optim.lr_scheduler import CosineAnnealingLR
from random import random
from torch.optim import Adam

from torch import (
    nn,
    load,
    float as pt_float,
    as_tensor,
    long,
    cat,
    ones,
    cumsum,
    arange,
)
from torch.utils.data import Dataset, DataLoader, random_split


device = "cuda" if is_available() else "cpu"

train, test = AG_NEWS(split=("train", "test"))
tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    """Tokenize."""
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
train, test = AG_NEWS(split=("train", "test"))  # iter
train = [
    (text, label - 1) for label, text in train
]  # originally starts at 1, so
test = [(text, label - 1) for label, text in test]


class DS(Dataset):
    def __init__(self, text, labels) -> None:
        self.text = text
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.text[idx]
        # X = text_transf(X)

        y = self.labels[idx]
        # y = label_transf(y)
        # y = as_tensor(y)

        return X, y


train_X, train_y = [text for text, _ in train], [label for _, label in train]
train_ds = DS(train_X, train_y)
test_X, test_y = [text for text, _ in test], [label for _, label in test]
test_ds = DS(test_X, test_y)

text_transf = lambda x: vocab(tokenizer(x))
label_transf = lambda x: int(x)


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(label_transf(_label))
        processed_text = as_tensor(text_transf(_text), dtype=long)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = as_tensor(label_list, dtype=long)
    offsets = as_tensor(offsets[:-1]).cumsum(dim=0)
    text_list = cat(text_list)
    return text_list, label_list, offsets


BATCH_SIZE = 64
train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    collate_fn=collate_batch,
    drop_last=True,
    num_workers=3,
)
# valid_dl = DataLoader(
#     valid_ds,
#     batch_size=BATCH_SIZE,
#     collate_fn=collate_batch,
#     drop_last=True,
#     num_workers=3,
# )
test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    collate_fn=collate_batch,
    drop_last=True,
    num_workers=3,
)


num_class = len(set([label for (text, label) in train]))
vocab_size = len(vocab)


class NET(nn.Module):
    def __init__(
        self,
        hidden_dim=12,
        embed_dim=12,
        act1="SELU",
        vocab_size=vocab_size,
        num_class=num_class,
    ):
        super().__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act1 = nn.__getattribute__(act1)()

        self.fc2 = nn.Linear(hidden_dim, num_class)
        # self.act2 = nn.__getattribute__(act2)()

        self.logs = nn.LogSoftmax(dim=-1)

        self.dropo = nn.Dropout(0.25)

    def count_weights_biases(self):
        return int(
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        )

    def forward(self, text, offsets):
        res = self.embedding(text, offsets)
        res = self.act1(self.dropo(self.fc1(res)))
        res = self.dropo(self.fc2(res))
        res = self.logs(res)
        return res


if __name__ == "__main__":

    EPOCHS = 20
    LR = 0.0005

    # Params
    net = NET().to(device).train()

    # Optimizer
    optimizer = Adam(net.parameters(), lr=LR)
    criterion = nn.NLLLoss()
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)
    ac = Accuracy(num_classes=4)
    f1 = F1(num_classes=4)

    # Train
    best_valid_acc = 0
    for epoch in range(EPOCHS):
        print(epoch, end=" | ")
        net = net.train()
        accuracy_scores_tr = []
        for i, (inputs, labels, offsets) in enumerate(train_dl):
            outputs = net(inputs.to(device), offsets.to(device))

            optimizer.zero_grad()
            loss = criterion(outputs, labels.to(device)).mean()
            loss.backward()
            # _ = clip_grad_norm_(net.parameters(), 0.25)
            optimizer.step()

            preds = outputs.argmax(dim=1)
            accuracy_scores_tr.append(f1(labels, preds.detach().cpu()))

            train_acc = stack(accuracy_scores_tr).mean()
            if random() > 0.95:
                print(f"{train_acc:^4.3f}", end=" ")
