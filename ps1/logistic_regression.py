import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

# Our input $x$
TEXT = NamedField(names=('seqlen',))

# Our labels $y$
LABEL = NamedField(sequential=False, names=(), unk_token=None)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=torch.device("cuda"))

batch = next(iter(train_iter))
print("Size of text batch:", batch.text.shape)
example = batch.text.get("batch", 1)
print("Second in batch", example)
print("Converted back to string:", " ".join([TEXT.vocab.itos[i] for i in example.tolist()]))

print("Size of label batch:", batch.label.shape)
example = batch.label.get("batch", 1)
print("Second in batch", example.item())
print("Converted back to string:", LABEL.vocab.itos[example.item()])

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

print("Word embeddings size ", TEXT.vocab.vectors.size())
print("Word embedding of 'follows', first 10 dim ", TEXT.vocab.vectors[TEXT.vocab.stoi['follows']][:10])

# def test_code(model):
#     "All models should be able to be run with following command."
#     upload = []
#     # Update: for kaggle the bucket iterator needs to have batch_size 10
#     test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
#     for batch in test_iter:
#         # Your prediction data here (don't cheat!)
#         probs = model(batch.text)
#         # here we assume that the name for dimension classes is `classes`
#         _, argmax = probs.max('classes')
#         upload += argmax.tolist()
#
#     with open("predictions.txt", "w") as f:
#         f.write("Id,Category\n")
#         for i, u in enumerate(upload):
#             f.write(str(i) + "," + str(u) + "\n")


# class baseModel(torch.nn.Module):
#     def __init__(self):
#         super(baseModel, self).__init__()

#
# class naiveBayesModel(nn.Module):
#     #uses binarized version
#     def __init__(self, dataset, alpha=1, vocabSize, batchSize):
#         super(baseModel, self).__init__()
#
#         self.vocabSize = vocabSize
#         #schematically:
#         N_p = 0
#         N_m = 0
#         p = ntorch.tensor(torch.ones(vocabSize, batchSize) * alpha, ['vocab', 'batch']) #batchsize?
#         q = ntorch.tensor(torch.ones(vocabSize, batchSize) * alpha, ['vocab', 'batch']) #batchsize?
#
#         for i, batch in enumerate(dataset):
#             if i%50==0: print(f"iteration {i}")
#             f = self.convertToX(batch)
#             #binarize f
#             f = torch.where(x > 0, torch.ones(f.size()), torch.zeros(f.size()))  # TODO
#
#
#             p += ntorch.where(batch.label==1., ones, zeros)
#             q += ntorch.where(batch.label==0., ones, zeros)
#
#
#             N_p = batch.label.sum("batch")
#
#             if batch.label == 'positive'  # TODO not correct
#                 N_p += 1
#                 p += batch.convert
#             else:
#                 N_m += 1
#                 q += f
#         r = torch.log( (p/torch.sum(p, 0)) / (q/torch.sum(q, 0)) )
#
#         self.W = r
#         self.b = ntorch.tensor(torch.log(N_p/N_m))  # TODO
#
#     def predict(self, x):
#         y = ntorch.tensor(torch.sign(self.W.index_select(x, 'vocab').sum('vocab') + self.b), ['classes', 'batch']) #TODO: sign function, mm
#         return y
#
#     def forward(self, text):
#         x = self.convertToX(batch)
#         return self.predict(x)
#
#     def convertToX(self, batch):
#         #this function makes the feature vectors wth scatter
#         x = ntorch.tensor( torch.zeros(vocabSize, batch.text.shape['batch']).cuda(), ('vocab', 'batch'))
#         y = ntorch.tensor( torch.ones(batch.text.shape['seqlen'], batch.text.shape['batch']).cuda(), ('seqlen', 'batch'))
#
#         x.scatter_('vocab', batch.text, y, 'seqlen')
#
#         print("len x:", len(x))
#         return x


class logisticRegression(torch.nn.Module):
    #uses binarized version
    def __init__(self, vocabSize, batchSize):
        super(logisticRegression, self).__init__()
        self.vocabSize = vocabSize

        self.W = ntorch.randn(self.vocabSize, requires_grad=True, names=("vocab",))
        self.b = ntorch.randn(self.vocabSize, requires_grad=True, names=("vocab",))

    # def train()
    #
    #     y.backward()

    def predict(self, x):
        y = (self.W.index_select(x, 'vocab').sum('vocab') + self.b).sigmoid()
        return y

    def forward(self, batch):
        x = self.convertToX(batch)
        return self.predict(x)

    def convertToX(self, batch):
        #this function makes the feature vectors wth scatter
        x = ntorch.tensor( torch.zeros(self.vocabSize, batch.text.shape['batch']).cuda(), ('vocab', 'batch'))
        y = ntorch.tensor( torch.ones(batch.text.shape['seqlen'], batch.text.shape['batch']).cuda(), ('seqlen', 'batch'))

        x.scatter_('vocab', batch.text, y, 'seqlen')

        print("len x:", len(x))
        return x


import ipdb; ipdb.set_trace()
