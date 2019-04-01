#model

import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

import args

#TODO add droppout

class MLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()
        self.l1 = ntorch.nn.Linear(in_size, in_size)
        self.l2 = ntorch.nn.Linear(in_size, out_size)

    def forward(self, x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x

    def spec(self, i_name, o_name):
        self.l1.spec(i_name, i_name)
        self.l2.spec(i_name, o_name)
        return self

class Model(nn.Module):
    def __init__(self, vectors):
        super(Model, self).__init__()

        self.embedding = ntorch.nn.Embedding(
                    vectors.shape[0] , args.embedding_size
                ).spec("embedding") #TODO is this good?

        self.embedding.weight.data.copy_(vectors)
        self.embedding.weight.requires_grad = False



        #can change these to MLPs if we want
        if args.intra_sentence:
            self.F_intra = ntorch.nn.Linear(args.embedding_size, args.embedding_size).spec("embedding", "embedding")
            self.F = ntorch.nn.Linear(2*args.embedding_size, args.f_out_size).spec("embedding", "embedding") 
        else:
            self.F = ntorch.nn.Linear(args.embedding_size, args.f_out_size).spec("embedding", "embedding") 
        self.G = ntorch.nn.Linear(2*args.f_out_size, args.g_out_size).spec("embedding", "embedding")
        self.H = ntorch.nn.Linear(2*args.g_out_size, args.n_classes).spec("embedding", "classes")



        self.lossfn = ntorch.nn.CrossEntropyLoss().spec("classes") 
        self.lossfn.reduction = None #TODO

    def forward(self, a, b):
        """
        The inputs are vectors, for now
        a: batch x seqlenA x embedding
        b: batch x seqlenB x embedding
        """
        a = self.embedding(a).rename("seqlen", "seqlenA")
        b = self.embedding(a).rename("seqlen", "seqlenB")

        if args.intra_sentence:
            #we ignore distance bias term because we are lazy
            a_p = a.dot("embedding", a).softmax("seqlenA").dot("seqlenA")
            b_p = b.dot("embedding", b).softmax("seqlenB").dot("seqlenB")

            a = ntorch.cat( (a, a_p), "embedding")
            b = ntorch.cat( (b, b_p), "embedding")

        a = self.F(a).relu()
        b = self.F(b).relu()

        alpha = a.dot("embedding", b).softmax("seqlenA").dot("seqlenA", a)
        beta = b.dot("embedding", a).softmax("seqlenB").dot("seqlenB", b)

        v1 = self.G(ntorch.cat( (a, beta), "embedding")).relu().sum("seqlenA")
        v2 = self.G(ntorch.cat( (b, alpha), "embedding")).relu().sum("seqlenB")

        y = self.H(ntorch.cat( (v1, v2), "embedding"))

        return y

    def loss(self, batch):

        y = self(a, b)
        loss = self.lossfn(y, tgt)
        return loss



