import torch
import torch.nn as nn
import torchtext
import copy
from torchtext.vocab import Vectors

from namedtensor import ntorch
from namedtensor.text import NamedField

import time
import numpy as np

from torchtext.data.iterator import BPTTIterator
from torchtext.data import Batch, Dataset
import math

bptt_len = 10
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
    TEXT.build_vocab(train, vectors="glove.840B.300d")
    vocab_size, embed_size = TEXT.vocab.vectors.size()

else:
    TEXT.build_vocab(train, max_size=20000)
    vocab_size = 20002
    embed_size = 60


train_iter, val_iter, test_iter = NamedBpttIterator.splits(
    (train, val, test), batch_size=batch_size, device=device, bptt_len=bptt_len, repeat=False)

class NNmodel(nn.Module):
    def __init__(self, embedding_size=60, hidden_size=128, vocab_size=1002, bptt_len=bptt_len, use_pretrained=False):
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
        out = self.embedding(batch)
        # out = self.dropout1(out)
        out = out.stack(("embedding", "seqlen"), "seqembedding")
        out = self.H(out).relu()
        # out = self.dropout2(out)
        out = self.W(out)

        return out

    def loss(self, batch):
        prediction = self(batch.text)
        next_words = batch.target.get('seqlen', batch.target.shape['seqlen'] - 1)
        # identity = ntorch.tensor(torch.diag(torch.ones(self.vocabSize)), ('index', 'vocab')).cuda()
        # one_hot_next_words = identity.index_select('index', next_words)

        return self.lossfn(prediction, next_words)


def test_code(model, name="sample.txt", lstm=False):
    #Cannot be the same code as in nn.py
        with open(name, "w") as fout:
        print("id,word", file=fout)
        for i, l in enumerate(open("input.txt"), 1):
            #w_2, w_1 = l.split(' ')[-3: -1]
            batch = [TEXT.vocab.stoi[word] for word in l.split(' ')[:-1]]
            batch = torch.tensor(batch).unsqueeze(1)
            batch = ntorch.tensor(batch, names=("seqlen", "batch")).cuda()
            #print(batch.shape)

            prediction_dist = model(batch).double()
            mask = np.zeros(vocab_size)
            mask[TEXT.vocab.stoi["<eos>"]] = float('-inf')
            mask[TEXT.vocab.stoi["<unk>"]] = float('-inf')
            mask[TEXT.vocab.stoi["<pad>"]] = float('-inf')

            torch_mask = ntorch.tensor(torch.from_numpy(mask), names=('vocab')).to(device=device)

            prediction_dist += torch_mask
            
            if lstm:
                prediction_dist = prediction_dist.get("seqlen", prediction_dist.shape['seqlen'] -1)
        

            top20 = prediction_dist.topk("vocab", 20)[1]

            predictions = [TEXT.vocab.itos[top20.get("vocab",i).item()] for i in range(20)]
            #print("predictions", predictions)

            print("%d,%s"%(i, " ".join(predictions)), file=fout)


if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()
    pretrained_embeddings = TEXT.vocab.vectors
    model = NNmodel(vocab_size=vocab_size,
                    embedding_size=embed_size,
                    use_pretrained=use_pretrained)
    #TODO use pretrained embeddings
    model.to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adadelta(parameters, lr=5.0)

    model_acc_list = []
    for epoch in range(5):
        tic = time.time()
        # eval_acc, sentence_vector = evaluate(model, x_test, y_test)
        model.train()
        losses = []
        for i, batch in enumerate(train_iter):
            if batch.text.shape['seqlen'] < bptt_len:
                continue
            optimizer.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        eval_ll = 0
        for i, batch in enumerate(val_iter):
            if batch.text.shape['seqlen'] < bptt_len:
                continue
            new_eval_ll = model.loss(batch)
            eval_ll += new_eval_ll.item()

        eval_ll /= (i + 1)

        print(
            "[epoch: {:d}] avg train_loss: {:.3f}   eval ll: {:.3f}   ({:.1f}s)".format(
                epoch, sum(losses)/len(losses), eval_ll, time.time() - tic
            )
        )
        model_copy = copy.deepcopy(model)
        model_acc_list.append([model_copy, eval_ll])


    #model.cpu()
    best_model, _ = min(model_acc_list, key=lambda x: x[1])
    best_model.eval()
    test_code(best_model)
