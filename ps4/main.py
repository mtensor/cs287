## TODO: main
import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

import args

# Our input $x$
TEXT = NamedField(names=('seqlen',))

# Our labels $y$
LABEL = NamedField(sequential=False, names=())

train, val, test = torchtext.datasets.SNLI.splits(
    TEXT, LABEL)

TEXT.build_vocab(train)
LABEL.build_vocab(train)
print('len(TEXT.vocab)', len(TEXT.vocab))
print('LABEL.vocab', LABEL.vocab)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=16, device=torch.device("cuda"), repeat=False)

import random
unk_vectors = [torch.randn(300) for _ in range(100)]
TEXT.vocab.load_vectors(vectors='glove.6B.300d',
                        unk_init=lambda x:random.choice(unk_vectors))

vectors = TEXT.vocab.vectors
vectors = vectors / vectors.norm(dim=1,keepdim=True)
ntorch_vectors = NamedTensor(vectors, ('word', 'embedding'))
TEXT.vocab.vectors = ntorch_vectors

def test_code(model, name="predictions.txt"):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10, device=torch.device("cuda"))
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.premise, batch.hypothesis)
        # here we assume that the name for dimension classes is `classes`
        _, argmax = probs.max('classes')
        upload += argmax.tolist()

    with open(name, "w") as f:
        f.write("Id,Category\n")
        for i, u in enumerate(upload):
            f.write(str(i) + "," + str(u) + "\n")

def train(model, debug=False):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    for epoch in range(8 if not debug else 1):
        tic = time.time()
        # eval_acc, sentence_vector = evaluate(model, x_test, y_test)
        model.train()
        losses = []
        for i, batch in enumerate(train_iter):
        	optimizer.zero_grad()
            a, b, y = batch.premise, batch.hypothesis, batch.label
            loss = model.loss(a, b, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if i%200==0:
                print(
                "\tavg train_loss: {:.3f} ({:.1f}s)".format(
                    sum(losses[-200:])/len(losses[-200:]), time.time() - tic
                )
            )

                import dill
                with open("model.p", 'rb') as h:
                    dill.dump(model, h)
                if debug: break

        model.eval()
        print("computing val loss ...")
        eval_ll = 0
        for i, batch in enumerate(val_iter):
            a, b, y = batch.premise, batch.hypothesis, batch.label
            new_eval_ll = model.loss(a, b, c)
            eval_ll += new_eval_ll.item()

        eval_ll /= (i + 1)

        print(
            "[epoch: {:d}] avg train_loss: {:.3f}   eval ll: {:.3f}   ({:.1f}s)".format(
                epoch, sum(losses)/len(losses), eval_ll, time.time() - tic
            )
        )

        print("running test code")
        name = "sample_"+ str(epoch) +".txt"
        test_code(model, name=name)

        print("ran test code")

if __name__ == '__main__':
	import args
	from model import Model
	model = Model(vectors).cuda()

	train(model)


