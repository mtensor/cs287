import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import Vectors

from namedtensor import ntorch
from namedtensor.text import NamedField

import time
import numpy as np

from torchtext.data.iterator import BPTTIterator
from torchtext.data import Batch, Dataset
import math

bptt_len = 5
mode = 'nonstatic'
device = torch.device("cuda")
use_pretrained = True
batch_size = 256
# Our input $x$
TEXT = NamedField(names=("seqlen",))


# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".",
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

# TEXT.build_vocab(train)
# print('len(TEXT.vocab)', len(TEXT.vocab))

if use_pretrained:
    TEXT.build_vocab(train, max_size=1000, vectors="glove.840B.300d")
    vocab_size, embed_size = TEXT.vocab.vectors.size()

else:
    TEXT.build_vocab(train, max_size=20000)
    vocab_size = 20002
    embed_size = 60


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
    (train, val, test), batch_size=batch_size, device=device, bptt_len=bptt_len, repeat=False)

class NNmodel(nn.Module):
    def __init__(self, embedding_size=60, hidden_size=512, vocab_size=1002, bptt_len=bptt_len, use_pretrained=False):
        super(NNmodel, self).__init__()
        self.vocabSize = vocab_size

        self.embedding = ntorch.nn.Embedding(
                    vocab_size, embedding_size
                ).spec("seqlen", "embedding")
        if use_pretrained:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = mode == "nonstatic"

        self.H = ntorch.nn.Linear(bptt_len * embedding_size, hidden_size).spec("seqembedding", "hidden")
        self.W = ntorch.nn.Linear(hidden_size, vocab_size).spec("hidden", "vocab")
        self.dropout1 = ntorch.nn.Dropout(0.9)
        self.dropout2 = ntorch.nn.Dropout(0.1)

        self.lossfn = ntorch.nn.CrossEntropyLoss().spec("vocab")

    def forward(self, batch):
        out = self.dropout1(self.embedding(batch))
        out = out.stack(("embedding", "seqlen"), "seqembedding")
        out = self.dropout2(self.H(out).relu())
        out = self.W(out)

        return out

    def loss(self, batch):
        prediction = self(batch.text)
        next_words = batch.target.get('seqlen', batch.target.shape['seqlen'] - 1)
        # identity = ntorch.tensor(torch.diag(torch.ones(self.vocabSize)), ('index', 'vocab')).cuda()
        # one_hot_next_words = identity.index_select('index', next_words)

        return self.lossfn(prediction, next_words)


def test_code(model):
    assert False

if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()
    pretrained_embeddings = TEXT.vocab.vectors
    model = NNmodel(vocab_size=vocab_size,
                    embedding_size=embed_size,
                    use_pretrained=use_pretrained)
    #TODO use pretrained embeddings
    model.to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    for epoch in range(20):
        tic = time.time()
        # eval_acc, sentence_vector = evaluate(model, x_test, y_test)
        model.train()
        losses = []
        for i, batch in enumerate(train_iter):
            if batch.text.shape['seqlen'] < 5:
                continue
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        eval_ll = 0
        for i, batch in enumerate(val_iter):
            if batch.text.shape['seqlen'] < 5:
                continue
            new_eval_ll = model.loss(batch)
            eval_ll += new_eval_ll.item()

        eval_ll /= (i + 1)

        print(
            "[epoch: {:d}] avg train_loss: {:.3f}   eval ll: {:.3f}   ({:.1f}s)".format(
                epoch, sum(losses)/len(losses), eval_ll, time.time() - tic
            )
        )

    #model.cpu()
    model.eval()
    test_code(model)
