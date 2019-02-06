#!pip install -q torch torchtext opt_einsum
#!pip install -U git+https://github.com/harvardnlp/namedtensor

import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe
import torch.nn as nn

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
import math


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

# batch = next(iter(train_iter))
# print("Size of text batch:", batch.text.shape)
# example = batch.text.get("batch", 1)
# print("Second in batch", example)
# print("Converted back to string:", " ".join([TEXT.vocab.itos[i] for i in example.tolist()]))

# print("Size of label batch:", batch.label.shape)
# example = batch.label.get("batch", 1)
# print("Second in batch", example.item())
# print("Converted back to string:", LABEL.vocab.itos[example.item()])

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

# print("Word embeddings size ", TEXT.vocab.vectors.size())
# print("Word embedding of 'follows', first 10 dim ", TEXT.vocab.vectors[TEXT.vocab.stoi['follows']][:10])

def test_code(model):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10, device=torch.device("cuda"))
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        # here we assume that the name for dimension classes is `classes`
        _, argmax = probs.max('classes')
        upload += argmax.tolist()

    with open("predictions.txt", "w") as f:
        f.write("Id,Category\n")
        for i, u in enumerate(upload):
            f.write(str(i) + "," + str(u) + "\n")

class baseModel(nn.Module):
	def __init__(self):
		super(baseModel, self).__init__()


class naiveBayesModel(nn.Module):
	#uses binarized version
	def __init__(self, dataset, vocabSize, batchSize, alpha=1):
		super(naiveBayesModel, self).__init__()


		self.vocabSize = vocabSize
		#schematically:
		N_p = 0
		N_m = 0
		p = ntorch.tensor(torch.ones(vocabSize) * alpha, ['vocab']).cuda() #batchsize?
		q = ntorch.tensor(torch.ones(vocabSize) * alpha, ['vocab']).cuda() #batchsize?

		ones = ntorch.tensor(torch.ones(vocabSize, batchSize), ['vocab', 'batch'])
		zeros = ntorch.tensor(torch.zeros(vocabSize, batchSize), ['vocab', 'batch'])

		for i, batch in enumerate(dataset):
			if i%100==0: print(f"iteration {i}")

			#gets binarized set-of-words
			f = self.convertToX(batch.text)
			#f = torch.where(x > 0, torch.ones(f.size()), torch.zeros(f.size()))  # TODO

			#p += ntorch.where(batch.label==1., ones, zeros).sum('batch')
			#q += ntorch.where(batch.label==0., ones, zeros).sum('batch')

			p += ntorch.dot("batch", batch.label.float(), f) #hopefully this works
			q += ntorch.dot("batch", (batch.label==0.).float(), f) #hopefully this works

			#print("q update:", (batch.label==0.).float())
			#print("p update:", batch.label.float())
			#assert False

			_n = batch.label.sum("batch").item()
			N_p += _n 
			N_m += batchSize - _n

		r = ntorch.log( (p / p.sum('vocab').item()) )  - ntorch.log(q / q.sum('vocab').item())   # TODO

		self.W = r
		self.b = ntorch.tensor(math.log(N_p / N_m), []).cuda()  # TODO

		print("b", self.b)
		print("W", self.W)

		print("N_p", N_p)
		print("N_m", N_m)
		print("sum W", self.W.sum("vocab"))
		assert False

	def predict(self, x):
		#y = ntorch.tensor(torch.sign(self.W.index_select(x, 'vocab').sum('vocab') + self.b), ['classes', 'batch']) #TODO: sign function, mm
		y_ = self.W.index_select('vocab', x.long()).sum('vocab').float() + self.b
		# tensor_a = ntorch.tensor(torch.Tensor([[1, 2], [3, 4]]), ("dim1", "dim2")
		# tensor_b = ntorch.tensor(torch.Tensor([[1, 2], [3, 4]]), ("dim1", "dim2"))
		# tensor_c = ntorch.stack([tensor_a, tensor_b], "dim3")
		
		print("y_", y_)
		y = ntorch.stack([y_>=0, y_<0], 'classes')

		return y

	def forward(self, text):
		x = self.convertToX(text)
		return self.predict(x)

	def convertToX(self, batchText):
		#this function makes the feature vectors wth scatter
		x = ntorch.tensor( torch.zeros(self.vocabSize, batchText.shape['batch']).cuda(), ('vocab', 'batch'))
		y = ntorch.tensor( torch.ones(batchText.shape['seqlen'], batchText.shape['batch']), ('seqlen', 'batch')).cuda()
		x.scatter_('vocab', batchText, y, 'seqlen')

		#print("x", x)
		#print(x.sum("vocab"))
	
		return x


if __name__=='__main__':
	model = naiveBayesModel(train_iter, len(TEXT.vocab), 10, alpha=1)

	test_code(model)

# #x = ntorch.tensor( torch.zeros(batch.text.shape['seqlen'],batch.text.shape['batch']), ('seqlen', 'batch'))
# x = ntorch.tensor( torch.zeros(vocabSize, batch.text.shape['batch']).cuda(), ('vocab', 'batch'))
# #y = torch.ones(batch.text.shape['seqlen'],batch.text.shape['batch'])
# y = ntorch.tensor( torch.ones(batch.text.shape['seqlen'],batch.text.shape['batch']).cuda(), ('seqlen', 'batch'))

# x.scatter_('vocab', batch.text, y, 'seqlen')




