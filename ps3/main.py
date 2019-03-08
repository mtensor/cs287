# Torch
import torch
# Text text processing library and methods for pretrained word embeddings
from torchtext import data, datasets
# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
import spacy

from seq2seq import Seq2Seq

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


BOS_WORD = '<s>'
EOS_WORD = '</s>'
DE = NamedField(names=('srcSeqlen',), tokenize=tokenize_de)
EN = NamedField(names=('trgSeqlen',), tokenize=tokenize_en,
                init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS

MAX_LEN = 20
import dill
import pickle
try:
    train, val = pickle.load(open("saved_data.p", 'rb'))
    print(loaded)
except: 
    print("could not load:")
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN),
                                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                           len(vars(x)['trg']) <= MAX_LEN)
    print("did filtering here")


MIN_FREQ = 5
DE.build_vocab(train.src, min_freq=MIN_FREQ)
EN.build_vocab(train.trg, min_freq=MIN_FREQ)
print(DE.vocab.freqs.most_common(10))
print("Size of German vocab", len(DE.vocab))
print(EN.vocab.freqs.most_common(10))
print("Size of English vocab", len(EN.vocab))
print(EN.vocab.stoi["<s>"], EN.vocab.stoi["</s>"])  # vocab index for <s>, </s>

BATCH_SIZE = 32
device = torch.device('cpu')#'cuda:0')
train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=device,
                                                  repeat=False, sort_key=lambda x: len(x.src))

batch = next(iter(train_iter))


def escape(l):
    return l.replace("\"", "<quote>").replace(",", "<comma>")

def kaggle_output(model):
    input_filename = "source_test.txt"

    with open(input_filename, 'rb') as file:
        for line in file:
            beam_list = model.beam_decode(line)
            trigrams = beam_to_trigrams(beam_list)

def beam_to_trigrams(beam_list):
    """ takes a list of the best 100 predictions (in order of decreasing likelihood)
    for a given sentence and outputs their initial trigrams in the kaggle format """
    assert len(beam_list) == 100  # we need 100 productions for eval
    trigrams = []
    for sentence in beam_list:
        trigrams.append('|'.join([EN.vocab.itos[i] for i in sentence[:3]]))

    return ' '.join(trigrams)



def train(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.001)

    for epoch in range(8 if not debug else 1):
        tic = time.time()
        # eval_acc, sentence_vector = evaluate(model, x_test, y_test)
        model.train()
        losses = []
        for i, batch in enumerate(train_iter):
            source, target = batch.src, batch.trg
            _, _, score = model(source, target)
            optimizer.zero_grad()
            loss = (-score).mean()
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
            source, target = batch.src, batch.trg
            _, _, new_eval_ll = model(source, target)
            eval_ll += new_eval_ll.item()

        eval_ll /= (i + 1)

        print(
            "[epoch: {:d}] avg train_loss: {:.3f}   eval ll: {:.3f}   ({:.1f}s)".format(
                epoch, sum(losses)/len(losses), eval_ll, time.time() - tic
            )
        )

        print("running test code")
        name = "sample_"+ str(epoch) +".txt"
        kaggle_output(model)

        print("ran test code")



if __name__ == '__main__':
	from seq2seq import Seq2Seq
	m = Seq2Seq(len(DE.vocab), len(EN.vocab))


	train(model)



