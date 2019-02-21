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

class LSTMmodel(nn.Module):
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


def test_code(model, name):
    #Cannot be the same code as in nn.py, because 
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
            
            prediction_dist = prediction_dist.get("seqlen", prediction_dist.shape['seqlen'] -1)
            #predictions = [TEXT.vocab.itos[i] for i in prediction_dist.topk("vocab", 20)[1] ]

            top20 = prediction_dist.topk("vocab", 20)[1]

            predictions = [TEXT.vocab.itos[top20.get("vocab",i).item()] for i in range(20)]
            #print("predictions", predictions)

            print("%d,%s"%(i, " ".join(predictions)), file=fout)

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
        test_code(model, name)

        print("ran test code")
    



