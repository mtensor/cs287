#!pip install -q torch torchtext opt_einsum
#!pip install -U git+https://github.com/harvardnlp/namedtensor
from __future__ import print_function, division

import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe
import torch.nn as nn

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
import math

import os
import sys
import time
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold

import data_helpers
from namedtensor import NamedTensor, ntorch


# Our input $x$
TEXT = NamedField(names=('seqlen',))

# Our labels $y$
LABEL = NamedField(sequential=False, names=(), unk_token=None)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

TEXT.build_vocab(train)
LABEL.build_vocab(train)

device=torch.device("cpu")

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=device)

batch = next(iter(train_iter))
print("Size of text batch:", batch.text.shape)
example = batch.text.get("batch", 1)
print("Second in batch", example)
print("Converted back to string:", " ".join([TEXT.vocab.itos[i] for i in example.tolist()]))

print("Size of label batch:", batch.label.shape)
example = batch.label.get("batch", 1)
print("Second in batch", example.item())
print("Converted back to string:", LABEL.vocab.itos[example.item()])

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

print("Word embeddings size ", TEXT.vocab.vectors.size())
print("Word embedding of 'follows', first 10 dim ", TEXT.vocab.vectors[TEXT.vocab.stoi['follows']][:10])

# batch = next(iter(train_iter))
# print("Size of text batch:", batch.text.shape)
# example = batch.text.get("batch", 1)
# print("Second in batch", example)
# print("Converted back to string:", " ".join([TEXT.vocab.itos[i] for i in example.tolist()]))

# print("Size of label batch:", batch.label.shape)
# example = batch.label.get("batch", 1)
# print("Second in batch", example.item())
# print("Converted back to string:", LABEL.vocab.itos[example.item()])

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

# print("Word embeddings size ", TEXT.vocab.vectors.size())
# print("Word embedding of 'follows', first 10 dim ", TEXT.vocab.vectors[TEXT.vocab.stoi['follows']][:10])

def test_code(model):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10, device=torch.device("cuda"))
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        # here we assume that the name for dimension classes is `classes`
        _, argmax = probs.max('classes')
        upload += argmax.tolist()

    with open("predictions.txt", "w") as f:
        f.write("Id,Category\n")
        for i, u in enumerate(upload):
            f.write(str(i) + "," + str(u) + "\n")


# for obtaining reproducible results
np.random.seed(0)
torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
print("use_cuda = {}\n".format(use_cuda))

# mode = "nonstatic"
mode = "static"
use_pretrained_embeddings = False
# use_pretrained_embeddings = True

num_classes = 2

# print("MODE      = {}".format(mode))
# print(
#     "EMBEDDING = {}\n".format(
#         "pretrained" if use_pretrained_embeddings else "random"
#     )
# )
#
# X, Y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
#
# vocab_size = len(vocabulary_inv_list)
# sentence_len = X.shape[1]
# num_classes = int(max(Y)) + 1  # added int() to convert np.int64 to int
#
# print("vocab size       = {}".format(vocab_size))
# print("max sentence len = {}".format(sentence_len))
# print("num of classes   = {}".format(num_classes))
#
# ConvMethod = "in_channel__is_embedding_dim"


class CNN(nn.Module):
    def __init__(
        self,
        kernel_sizes=[3, 4, 5],
        num_filters=100,
        embedding_dim=300,
        pretrained_embeddings=None,
    ):
        super(CNN, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.embedding = ntorch.nn.Embedding(
            vocab_size, embedding_dim
        ).augment("h")
        self.embedding.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings)
        )
        self.embedding.weight.requires_grad = mode == "nonstatic"

        conv_blocks = []
        for kernel_size in kernel_sizes:
            conv1d = ntorch.nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=1,
            )

            conv_blocks.append(conv1d)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.fc = ntorch.nn.Linear(
            num_filters * len(kernel_sizes), num_classes
        ).spec("h", "classes")
        self.dropout = ntorch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).transpose("h", "seqlen")
        x_list = [
            conv_block(x).relu().max("seqlen")[0]
            for conv_block in self.conv_blocks
        ]
        out = ntorch.cat(x_list, "h")
        feature_extracted = out
        out = self.fc(self.dropout(out)).softmax("classes")
        return out, feature_extracted


def evaluate(model, x_test, y_test):
    # inputs = NamedTensor(x_test, ("batch", "seqlen"))
    # y_test = NamedTensor(y_test, ("batch",))
    preds, vector = model(x_test)
    preds = preds.max("classes")[1]
    eval_acc = (preds == y_test).sum("batch").item() / len(y_test)
    return eval_acc, vector.cpu().detach().numpy()


embedding_dim = 300
num_filters = 100
kernel_sizes = [3, 4, 5]
batch_size = 50
vocab_size = TEXT.vocab.vectors.shape[0]


if use_pretrained_embeddings:
    pretrained_embeddings = None
else:
    pretrained_embeddings = np.random.uniform(
        -0.01, -0.01, size=(vocab_size, embedding_dim)
    )


def train_test_one_split():
    model = CNN(pretrained_embeddings=pretrained_embeddings)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.01)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        tic = time.time()
        # eval_acc, sentence_vector = evaluate(model, x_test, y_test)
        model.train()
        for i, (data, _) in enumerate(iter(train_iter)):
            inputs, labels = data
            # inputs = NamedTensor(inputs, ("batch", "seqlen"))
            # labels = NamedTensor(labels, ("batch",))
            preds, _ = model(inputs)

            loss = preds.reduce2(labels, loss_fn, ("batch", "classes"))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        eval_acc = 0
        for i, (data, _) in enumerate(iter(val_iter)):
            x_test, y_test = data
            new_eval_acc, sentence_vector = evaluate(model, x_test, y_test)
            eval_acc += new_eval_acc

        eval_acc /= (i + 1)

        print(
            "[epoch: {:d}] train_loss: {:.3f}   acc: {:.3f}   ({:.1f}s)".format(
                epoch, loss.item(), eval_acc, time.time() - tic
            )
        )
    return eval_acc, sentence_vector


def do_cnn():
    cv_folds = 10
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
    acc_list = []
    tic = time.time()
    sentence_vectors, y_tests = [], []
    for cv, (train_index, test_index) in enumerate(kf.split(X)):
        acc, sentence_vec = train_test_one_split(cv, train_index, test_index)
        print(
            "cv = {}    train size = {}    test size = {}\n".format(
                cv, len(train_index), len(test_index)
            )
        )
        acc_list.append(acc)
        sentence_vectors += sentence_vec.tolist()
        y_tests += Y[test_index].tolist()
    print(
        "\navg acc = {:.3f}   (total time: {:.1f}s)\n".format(
            sum(acc_list) / len(acc_list), time.time() - tic
        )
    )

    np.save("models/sentence_vectors.npy", np.array(sentence_vectors))
    np.save("models/sentence_vectors_y.npy", np.array(y_tests))


if __name__ == "__main__":
    train_test_one_split()
