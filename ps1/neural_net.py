import torch
from torch import optim
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

device=torch.device("cpu")

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10, device=device)



# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

def test_code(model):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
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


class CBOW(ntorch.nn.Module):
    #uses binarized version
    def __init__(self, vocabSize, embedSize):
        super(CBOW, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.Vsize = 1000

        # self.Wb = ntorch.nn.Linear(self.embedSize, 1)
        #self.W = Variable ntorch.tensor(torch.zeros((self.vocabSize), requires_grad=True), ("vocab",)))
        self.V = ntorch.nn.Linear(self.embedSize, self.Vsize).spec("vocab", "embedding")
        self.U = ntorch.nn.Linear(self.Vsize, 1).spec("embedding", "score")
        # self.relu = ntorch.nn.ReLU().spec("singular", "singular")
        #self.b = ntorch.tensor(0., names=())
        self.lossfn = ntorch.nn.CrossEntropyLoss().spec("classes")

    def predict(self, x):
        # import ipdb; ipdb.set_trace()
        # y = self.Wb
        #y = (self.W.index_select('vocab', x.long()).sum('vocab') + self.b).sigmoid()
        y_ = self.U(self.V(x).relu().sum("seqlen")).sigmoid().sum('score')
        # y_ = self.relu(self.W(x)).sigmoid().sum('singular') # this is a huge hack

        y = ntorch.stack([y_, 1-y_], 'classes') #.log_softmax('classes')
        return y

    def forward(self, batchText):
        x = self.convertToX(batchText)
        return self.predict(x)

    def convertToX(self, batchText):
        # import ipdb; ipdb.set_trace()
        #this function makes the feature vectors wth scatter
        # x = ntorch.tensor( torch.zeros(self.vocabSize, batchText.shape['batch'], device=device), ('vocab', 'batch'))
        # y = ntorch.tensor( torch.ones(batchText.shape['seqlen'], batchText.shape['batch'], device=device), ('seqlen', 'batch'))
        # one_hot_vectors = ntorch.tensor(torch.diag(torch.ones(self.vocabSize)), ('vocab', 'lookup'))
        pretrained_embeddings = ntorch.tensor(TEXT.vocab.vectors, ('lookup', 'vocab'))
        x = pretrained_embeddings.index_select('lookup', batchText)
        # x.scatter_('vocab', batchText, y, 'seqlen')
        #print("len x:", len(x))
        return x

    def loss(self, batch):
        prediction = self(batch.text)  # probabilities
        #print("predict", prediction)
        #print('predict size', prediction.shape)
        #print("label", batch.label)
        return self.lossfn(prediction, batch.label)

def train(cbow, dataset):
    optimizer = optim.Adam(cbow.parameters(), lr=0.0001)

    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    losses = []
    for i, batch in enumerate(dataset):
        optimizer.zero_grad()   # zero the gradient buffers

        loss = cbow.loss(batch) #loss = (batch.label.float() - prediction).abs().sum('batch')
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i%200==0 and not i==0:
            print(f"iteration {i}")
            print(f"moving average loss={sum(losses[-1-100:-1])/100.}")
            val_losses = []
            for vbatch in val_iter:
                val_losses.append( cbow.loss(vbatch).item())
            val_loss = sum(val_losses)/len(val_losses)
            print(f"val loss: {val_loss}")
#import ipdb; ipdb.set_trace()

if __name__=='__main__':
#print("params", lr.parameters())
    cbow = CBOW(TEXT.vocab.vectors.size()[0], TEXT.vocab.vectors.size()[1])

    for i in range(10):
        print(f"epoch {i}")
        train(cbow, train_iter)


    test_code(cbow)
