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


class logisticRegression(ntorch.nn.Module):
    #uses binarized version
    def __init__(self, vocabSize, embedSize):
        super(logisticRegression, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize

        # self.Wb = ntorch.nn.Linear(self.embedSize, 1)
        #self.W = Variable ntorch.tensor(torch.zeros((self.vocabSize), requires_grad=True), ("vocab",)))
        self.W = ntorch.nn.Linear(self.vocabSize, 1).spec("vocab", "singular")
        #self.b = ntorch.tensor(0., names=())
        self.lossfn = ntorch.nn.CrossEntropyLoss().spec("classes")

    def predict(self, x):
        # y = self.Wb
        #y = (self.W.index_select('vocab', x.long()).sum('vocab') + self.b).sigmoid()
        y_ = self.W(x).sigmoid().sum('singular') # this is a huge hack

        y = ntorch.stack([y_, 1-y_], 'classes') #.log_softmax('classes')
        return y

    def forward(self, batchText):
        x = self.convertToX(batchText)
        return self.predict(x)

    def convertToX(self, batchText):
        #this function makes the feature vectors wth scatter
        x = ntorch.tensor( torch.zeros(self.vocabSize, batchText.shape['batch'], device=device), ('vocab', 'batch'))
        y = ntorch.tensor( torch.ones(batchText.shape['seqlen'], batchText.shape['batch'], device=device), ('seqlen', 'batch'))

        x.scatter_('vocab', batchText, y, 'seqlen')
        #print("len x:", len(x))
        return x

    def loss(self, batch):
        prediction = self(batch.text)  # probabilities
        #print("predict", prediction)
        #print('predict size', prediction.shape)
        #print("label", batch.label)
        return self.lossfn(prediction, batch.label)

def train(lrModel, dataset):
    optimizer = optim.Adam(lrModel.parameters(), lr=0.01)

    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    losses = []
    for i, batch in enumerate(dataset):
        optimizer.zero_grad()   # zero the gradient buffers
        
        loss = lrModel.loss(batch) #loss = (batch.label.float() - prediction).abs().sum('batch')
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if i%200==0 and not i==0:
            print(f"iteration {i}")
            print(f"moving average loss={sum(losses[-1-100:-1])/100.}")
            val_losses = []
            for vbatch in val_iter:
                val_losses.append( lrModel.loss(vbatch).item())
            val_loss = sum(val_losses)/len(val_losses)
            print(f"val loss: {val_loss}")
#import ipdb; ipdb.set_trace()

if __name__=='__main__':
#print("params", lr.parameters())
    lrModel = logisticRegression(TEXT.vocab.vectors.size()[0], None)

    for i in range(10):
        print(f"epoch {i}")
        train(lrModel, train_iter)


    test_code(lrModel)


