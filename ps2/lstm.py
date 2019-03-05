import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import Vectors

from namedtensor import ntorch
from namedtensor.text import NamedField

import numpy as np

from torchtext.data.iterator import BPTTIterator
from torchtext.data import Batch, Dataset
import math
import time

#Our imports
from trigram import NamedBpttIterator
from nn import test_code

use_pretrained = True
mode = 'nonstatic'
device = torch.device("cuda")

# Our input $x$
TEXT = NamedField(names=("seqlen",))

# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".",
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

if use_pretrained:
    TEXT.build_vocab(train, vectors="glove.840B.300d")
    vocab_size, embed_size = TEXT.vocab.vectors.size()
    print("embedding size", embed_size)

else:
    TEXT.build_vocab(train)
    vocab_size = 1002
    embed_size = 128

train_iter, val_iter, test_iter = NamedBpttIterator.splits(
    (train, val, test), batch_size=10, device=torch.device("cuda"), bptt_len=32, repeat=False)

class LSTMmodel(torch.nn.Module):
    def __init__(self, embedding_size=embed_size, hidden_size=512, num_layers=2, vocab_size=10001, use_pretrained=False, dropout=0.3):
        super(LSTMmodel, self).__init__()

        
        self.embedding = ntorch.nn.Embedding(
                    vocab_size, embedding_size
                )
        if use_pretrained:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = mode == "nonstatic"
        
        #decoder, outputs to "vocab"

        #(seq_len, batch_size, features)
        self.lstm = ntorch.nn.LSTM(embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False,
                            dropout=dropout
                            ).spec("embedding", "seqlen", "output")

        self.decoder = ntorch.nn.Linear(
            hidden_size, vocab_size
        ).spec("output", "vocab")

        self.lossfn = ntorch.nn.CrossEntropyLoss().spec("vocab") 

    def forward(self, batch):
        #print("batch:", batch)
        batch = self.embedding(batch)

        out, _ = self.lstm(batch)

        dist = self.decoder(out)
        #out is a dist over vocab for the batches, log likelihood space 
        return dist

    def loss(self, batch):
        prediction = self(batch.text)  # probabilities
        return self.lossfn(prediction, batch.target)


if __name__ == "__main__":

    debug = False
    pretrained_embeddings = TEXT.vocab.vectors
    model = LSTMmodel(embedding_size=embed_size, use_pretrained=use_pretrained, dropout=0.5) #TODO
    model.cuda()
    #model.cpu()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    for epoch in range(8 if not debug else 1):
        tic = time.time()
        # eval_acc, sentence_vector = evaluate(model, x_test, y_test)
        model.train()
        losses = []
        for i, batch in enumerate(train_iter):
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if i%200==0:
                print(
                "\tavg train_loss: {:.3f} ({:.1f}s)".format(
                    sum(losses[-200:])/len(losses[-200:]), time.time() - tic
                )
            )
                if debug: break

        model.eval()
        print("computing val loss ...")
        eval_ll = 0
        for i, batch in enumerate(val_iter):
            new_eval_ll = model.loss(batch)
            eval_ll += new_eval_ll.item()

        eval_ll /= (i + 1)

        print(
            "[epoch: {:d}] avg train_loss: {:.3f}   eval ll: {:.3f}   ({:.1f}s)".format(
                epoch, sum(losses)/len(losses), eval_ll, time.time() - tic
            )
        )

        print("running test code")
        name = "sample_"+ str(epoch) +".txt"
        test_code(model, name=name, lstm=True)

        print("ran test code")
    



