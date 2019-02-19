import torch
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

print('len(train)', len(train))

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

it = iter(train_iter)
batch = next(it)
print("Size of text batch [max bptt length, batch size]", batch.text.shape)
example = batch.text[{"batch": 1}]
print("Second in batch", example)
print("Converted back to string: ", " ".join([TEXT.vocab.itos[i] for i in example.values.data]))

import ipdb; ipdb.set_trace()
