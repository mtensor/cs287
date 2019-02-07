import torch
from torch import optim
# Text text processing library and methods for pretrained word embeddings
import torchtext
import copy
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

device=torch.device("cuda")

# Our input $x$
TEXT = NamedField(names=('seqlen',))

# Our labels $y$
LABEL = NamedField(sequential=False, names=(), unk_token=None)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=64, device=device)

# TEXT.vocab.load_vectors()
TEXT.build_vocab(train, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train)



# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'

# TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

def test_code(model):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10, device=device)
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


class CNN(ntorch.nn.Module):
    #uses binarized version
    def __init__(self, vocabSize, embedSize):
        super(CNN, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.kernel_sizes = [3, 4, 5]
        self.n_filters = 200

        # self.Wb = ntorch.nn.Linear(self.embedSize, 1)
        #self.W = Variable ntorch.tensor(torch.zeros((self.vocabSize), requires_grad=True), ("vocab",)))
        self.embedding = ntorch.nn.Embedding(self.vocabSize, self.embedSize).spec("seqlen", "h")
        # self.V = ntorch.nn.Linear(self.embedSize, self.Vsize).spec("vocab", "embedding")
        # self.U = ntorch.nn.Linear(self.embedSize, 2).spec("embedding", "classes")
        self.convs = torch.nn.ModuleList([ntorch.nn.Conv1d(
            in_channels=self.embedSize,
            out_channels=self.n_filters,
            kernel_size=(kernel_size),
            padding=1)
            for kernel_size in self.kernel_sizes])
        self.fc = ntorch.nn.Linear(
            self.n_filters * len(self.kernel_sizes), 2
            ).spec("h", "classes")
        self.dropout = ntorch.nn.Dropout(0.5)
        # self.relu = ntorch.nn.ReLU().spec("singular", "singular")
        #self.b = ntorch.tensor(0., names=())
        self.lossfn = ntorch.nn.CrossEntropyLoss().spec("classes")

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        # y = self.Wb
        #y = (self.W.index_select('vocab', x.long()).sum('vocab') + self.b).sigmoid()
        # y_ = self.U(self.dropout(self.V(x)).relu().sum("seqlen")).sum('score').sigmoid()
        x = self.embedding(x).transpose("h", "seqlen")
        x_list = [
            conv_block(x).relu().max("seqlen")[0]
            for conv_block in self.convs
        ]
        # y_ = self.U self.embedding(x).sum("seqlen")
        out = ntorch.cat(x_list, "h")
        # feature_extracted = out
        out = self.fc(self.dropout(out)).softmax("classes")
        # y_ = self.relu(self.W(x)).sigmoid().sum('singular') # this is a huge hack

        # y = ntorch.stack([y_, 1.-y_], 'classes') #.log_softmax('classes')
        return out

    def loss(self, batch):
        prediction = self(batch.text)  # probabilities
        #print("predict", prediction)
        #print('predict size', prediction.shape)
        #print("label", batch.label)
        return self.lossfn(prediction, batch.label)

    def acc(self, batch):
        prediction = self(batch.text)  # probabilities
        acc = (batch.label == prediction.argmax("classes")).float().mean("batch")

        #print("predict", prediction)
        #print('predict size', prediction.shape)
        #print("label", batch.label)
        return acc

def train(cnn, dataset):
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    # in your training loop:
    for i, batch in enumerate(dataset):
        cnn.train()
        optimizer.zero_grad()   # zero the gradient buffers

        loss = cnn.loss(batch) #loss = (batch.label.float() - prediction).abs().sum('batch')
        loss.backward()
        optimizer.step()
        if i%100==0 and not i==0:
        # if True:
            cnn.eval()
            train_losses = []
            train_accs = []
            for tbatch in train_iter:
                train_losses.append( cnn.loss(tbatch).item())
                train_accs.append( cnn.acc(tbatch).item())
            train_loss = sum(train_losses)/len(train_losses)
            train_accs = sum(train_accs)/len(train_accs)
            print(f"train loss: {train_loss}")
            print(f"train acc: {train_accs}")
            print(f"iteration {i}")
            # print(f"moving average loss={sum(losses[-1-100:-1])/100.}")
            val_losses = []
            val_accs = []
            for vbatch in val_iter:
                val_losses.append( cnn.loss(vbatch).item())
                val_accs.append( cnn.acc(vbatch).item())
            val_loss = sum(val_losses)/len(val_losses)
            val_accs = sum(val_accs)/len(val_accs)
            print(f"val loss: {val_loss}")
            print(f"val acc: {val_accs}")

    return cnn, val_accs
#import ipdb; ipdb.set_trace()

if __name__=='__main__':
#print("params", lr.parameters())
    cnn = CNN(TEXT.vocab.vectors.size()[0], TEXT.vocab.vectors.size()[1])
    cnn.to(device)
    pretrained_embeddings = TEXT.vocab.vectors
    cnn.embedding.weight.data.copy_(pretrained_embeddings)

    model_acc_list = []
    for i in range(20):
        print(f"epoch {i}")
        curr_model, val_accs = train(cnn, train_iter)
        model_copy = copy.deepcopy(curr_model)
        model_acc_list.append([model_copy, val_accs])

    best_model, _ = max(model_acc_list, key=lambda x: x[1])
    test_code(best_model)
