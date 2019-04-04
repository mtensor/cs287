#model

import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

import args

import torch.nn as nn
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
                ).spec("seqlen", "embedding") #TODO is this good?

        self.embedding.weight.data.copy_(vectors)
        self.embedding.weight.requires_grad = False



        #can change these to MLPs if we want
        if args.intra_sentence:
            self.F_intra = MLP(args.embedding_size, args.embedding_size).spec("embedding", "embedding")
            self.F = MLP(2*args.embedding_size, args.f_out_size).spec("embedding", "embedding") 
        else:
            self.F = MLP(args.embedding_size, args.f_out_size).spec("embedding", "embedding") 
        self.G = MLP(2*args.f_out_size, args.g_out_size).spec("embedding", "embedding")
        self.H = MLP(2*args.g_out_size, args.n_classes).spec("embedding", "classes")



        self.lossfn = ntorch.nn.CrossEntropyLoss().spec("classes") 
        self.lossfn.reduction = "none" #TODO

    def forward(self, a, b, show_attn=False):
        """
        The inputs are vectors, for now
        a: batch x seqlenA x embedding
        b: batch x seqlenB x embedding
        """
        a = self.embedding(a).rename("seqlen", "seqlenA")
        b = self.embedding(b).rename("seqlen", "seqlenB")

        if args.intra_sentence:
            #we ignore distance bias term because we are lazy
            a_p = a.dot("embedding", a.rename("seqlenA", "sl")).softmax("sl").dot("sl", a.rename("seqlenA", "sl"))
            b_p = b.dot("embedding", b.rename("seqlenB", "sl")).softmax("sl").dot("sl", b.rename("seqlenB", "sl"))

            a = ntorch.cat( (a, a_p), "embedding")
            b = ntorch.cat( (b, b_p), "embedding")

        a = self.F(a).relu()
        b = self.F(b).relu()


        attns_alpha = a.dot("embedding", b).softmax("seqlenA")
        attns_beta = b.dot("embedding", a).softmax("seqlenB")
        if show_attn: return attns_alpha, attns_beta

        alpha = attns_alpha.dot("seqlenA", a)
        beta = attns_beta.dot("seqlenB", b)

        v1 = self.G(ntorch.cat( (a, beta), "embedding")).relu().sum("seqlenA")
        v2 = self.G(ntorch.cat( (b, alpha), "embedding")).relu().sum("seqlenB")

        y = self.H(ntorch.cat( (v1, v2), "embedding"))

        return y

    def loss(self, a, b, tgt):

        y = self(a, b)
        loss = self.lossfn(y, tgt)

        return loss.mean("batch")


    def showAttention(self, a, b, TEXT):
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        import matplotlib.ticker as ticker
        
        attns_alpha, attns_beta = self(a, b, show_attn=True)


        input_words = [ TEXT.vocab.itos[a[{'batch':0, 'seqlen':i}].item()] for i in range(a.shape['seqlen'])  ]
        print('input words', input_words)        

        output_words = [ TEXT.vocab.itos[b[{'batch':0, 'seqlen':i}].item()] for i in range(b.shape['seqlen'])  ]

        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attns_alpha[{'batch':0}].detach().numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_words +
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
        fig.savefig('attnA.png')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attns_beta[{'batch':0}].detach().numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_words + #switch?
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()
        fig.savefig('attnB.png')

