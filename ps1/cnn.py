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

#from sklearn.model_selection import KFold

#import data_helpers
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

device=torch.device("cuda")

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=device)

url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))


def test_code(model):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10, device=torch.device("cpu"))
    for batch in test_iter:
        #print("SEQLEN:", batch.text.shape)
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
use_pretrained_embeddings = True
print(f"use_pretrained_embeddings: {use_pretrained_embeddings}")
print(f"embedding mode: {mode}")
num_classes = 2


class CNN(nn.Module):
    def __init__(
        self,
        kernel_sizes=[3, 4, 5],
        num_filters=32,
        embedding_dim=300,
        pretrained_embeddings=None,
    ):
        super(CNN, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.embedding = ntorch.nn.Embedding(
            vocab_size, embedding_dim
        ).augment("h")
        self.embedding.weight.data.copy_(pretrained_embeddings)
        #self.embedding.cuda()
        #self.embedding.weight = self.embedding.weight.cuda()
        self.embedding.weight.requires_grad = mode == "nonstatic"

        conv_blocks = []
        for kernel_size in kernel_sizes:
            conv1d = ntorch.nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            ).spec("h", "seqlen", "features")

            conv_blocks.append(conv1d)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.fc = ntorch.nn.Linear(
            num_filters * len(kernel_sizes), 2
        ).spec("features", "classes")
        self.dropout = ntorch.nn.Dropout(0.8)

        self.lossfn = ntorch.nn.NLLLoss().spec("classes")

    def forward(self, x):
        #print("x shape1", x.shape)
        x = self.embedding(x).transpose("h", "seqlen")
        #print("x shape2", x.shape)
        x_list = [
            conv_block(x).relu().max("seqlen")[0]
            for conv_block in self.conv_blocks
            ]
        out = ntorch.cat(x_list, "features")
        #print("out shape", out.shape)
        feature_extracted = out
        out = self.fc(self.dropout(out)).log_softmax("classes")
        #print("out2 shape", out.shape)
        #assert False
        return out

    def loss(self, batch):
        prediction = self(batch.text)  # probabilities
        return self.lossfn(prediction, batch.label)


# embedding_dim = 300
# num_filters = 32
# kernel_sizes = [3, 4, 3]
vocab_size = TEXT.vocab.vectors.shape[0]

if use_pretrained_embeddings:
    pretrained_embeddings = TEXT.vocab.vectors
else:
    pretrained_embeddings = torch.from_numpy(np.random.uniform(
        -0.01, -0.01, size=(vocab_size, embedding_dim)
    ))

if __name__ == "__main__":

    model = CNN(pretrained_embeddings=pretrained_embeddings)
    model.cuda()

    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # for i in parameters:
    #     print("param:", i)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    loss_fn = nn.NLLLoss()

    for epoch in range(20):
        tic = time.time()
        # eval_acc, sentence_vector = evaluate(model, x_test, y_test)
        model.train()

        #batch = next(iter(train_iter))
        #i = 0
        losses = []
        # while i < 200:
        #     i += 1
        for i, batch in enumerate(train_iter):

            #preds, _ = model(batch.text)
            #loss = preds.reduce2(labels, loss_fn, ("batch", "classes"))
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())


        model.eval()
        eval_ll = 0
        for i, batch in enumerate(val_iter):
            #new_eval_acc, sentence_vector = evaluate(model, x_test, y_test)
            #new_eval_ll = model(x_test)[0].reduce2(y_test, loss_fn, ("batch", "classes")).item()
            new_eval_ll = model.loss(batch)
            eval_ll += new_eval_ll.item()

        eval_ll /= (i + 1)

        print(
            "[epoch: {:d}] avg train_loss: {:.3f}   eval ll: {:.3f}   ({:.1f}s)".format(
                epoch, sum(losses)/len(losses), eval_ll, time.time() - tic
            )
        )

    model.cpu()
    model.eval()
    test_code(model)