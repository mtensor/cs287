import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

import args

import torch.nn as nn

from model import Model


class Ensemble(nn.Module):
    def __init__(self, vectors, K=10):
        super(Ensemble, self).__init__()
        self.models = [Model(vectors) for model in range(K)]
        self.lossfn = ntorch.nn.CrossEntropyLoss().spec("classes")
        self.lossfn.reduction = "none" #TODO

    def forward(self, a, b):
        """
        The inputs are vectors, for now
        a: batch x seqlenA x embedding
        b: batch x seqlenB x embedding
        """
        y = ntorch.stack([model(a, b) for model in self.models],
                         name='ensemble').mean('ensemble')

        return y

    def loss(self, a, b, tgt):

        y = self(a, b)
        loss = self.lossfn(y, tgt)

        return loss.mean("batch")
