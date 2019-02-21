import torch
import torchtext
from torchtext.vocab import Vectors
from collections import Counter

from namedtensor import ntorch
from namedtensor.text import NamedField

import numpy as np

from torchtext.data.iterator import BPTTIterator
from torchtext.data import Batch, Dataset
import math

from itertools import islice

import nltk
from nltk.corpus import brown


# Our input $x$
TEXT = NamedField(names=("seqlen",))

# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=".",
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)

print('len(train)', len(train))

TEXT.build_vocab(train)
print('len(TEXT.vocab)', len(TEXT.vocab))

if False:
    TEXT.build_vocab(train, max_size=1000)
    len(TEXT.vocab)

nltk.download('brown')
brown_1gram = nltk.FreqDist(brown.words())
brown_2gram = nltk.ConditionalFreqDist(nltk.bigrams(brown.words()))
brown_trigrams = nltk.trigrams(brown.words())
condition_pairs = (((w0, w1), w2) for w0, w1, w2 in brown_trigrams)
brown_3gram = nltk.ConditionalFreqDist(condition_pairs)


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

unigram_count, bigram_count, trigram_count = Counter(), Counter(), Counter()

for b in iter(train_iter):
    words = b.text.values.contiguous().view(-1).tolist()
    unigram_count.update(words)
    bigram_count.update(zip(words, words[1:]))
    trigram_count.update(zip(words, words[1:], words[2:]))


with open("sample.txt", "w") as fout:
    print("id,word", file=fout)
    for i, l in enumerate(open("input.txt"), 1):
        w_2, w_1 = l.split(' ')[-3: -1]
        # w_2, w_1 = TEXT.vocab.stoi[w_2], TEXT.vocab.stoi[w_1]
        prediction_dict = Counter()
        for word in range(len(TEXT.vocab)):
            word = TEXT.vocab.itos[word]
            unigram_score = brown_1gram[word]
            bigram_score = brown_2gram[w_1][word]
            trigram_score = brown_3gram[(w_2, w_1)][word]
            # prediction_dict[word] = trigram_score
            prediction_dict[word] = (1e20 * trigram_score) + (1e10 * bigram_score) + unigram_score
        prediction_dict["<eos>"] = 0
        prediction_dict["<unk>"] = 0
        prediction_dict["<pad>"] = 0
        predictions = [i for i, c in prediction_dict.most_common(20)]

        print("%d,%s"%(i, " ".join(predictions)), file=fout)
