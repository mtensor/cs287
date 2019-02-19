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


# Our input $x$
TEXT = NamedField(names=("seqlen",))

# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".",
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

#print('len(train)', len(train))

# TEXT.build_vocab(train)
# print('len(TEXT.vocab)', len(TEXT.vocab))

if True:
    TEXT.build_vocab(train, max_size=1000)
    len(TEXT.vocab)


class NamedBpttIterator(BPTTIterator):
    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size)
                                              * self.batch_size - len(text)))
        data = TEXT.numericalize(
            [text], device=self.device)
        data = (data
            .stack(("seqlen", "batch"), "flat")
            .split("flat", ("batch", "seqlen"), batch=self.batch_size)
            .transpose("seqlen", "batch")
        )

        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('target', TEXT)])
        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                seq_len = min(self.bptt_len, len(data) - i - 1)
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text = data.narrow("seqlen", i, seq_len),
                    target = data.narrow("seqlen", i+1, seq_len),
                )

            if not self.repeat:
                return


train_iter, val_iter, test_iter = NamedBpttIterator.splits(
    (train, val, test), batch_size=10, device=torch.device("cuda"), bptt_len=32, repeat=False)

LSTMmodel(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=512, num_layers=1):
        super(LSTMmodel, self).__init__()


            #encoder
            #decoder, outputs to "vocab"
            #(seq_len, batch_size, features)
        self.lstm = nn.LSTM(embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False,
                            dropout = 0.3
                            ).spec("input", "seqlen", "output")

        self.lossfn = ntorch.nn.CrossEntropyLoss().spec("vocab") 

    def forward(self, batch):

        out, hiddens = self.lstm(batch)

        dist = self.decoder(out)
        #out is a dist over vocab for the batches, log likelihood space 


    def loss(self, batch):
        prediction = self(batch.text)  # probabilities
        return self.lossfn(prediction, batch.target)


def test_code(model):
    

if __name__ == "__main__":

    model = LSTMmodel(pretrained_embeddings=pretrained_embeddings) #TODO
    model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    for epoch in range(20):
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

        model.eval()
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

    model.cpu()
    model.eval()
    test_code(model)



