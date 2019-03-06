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
import pickle
try:
    train, val = pickle.load(open("saved_data.p", 'rb'))
    print(loaded)
except: 
    print("could not load:")
    train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN),
                                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                                           len(vars(x)['trg']) <= MAX_LEN)
    with open("saved_data.p", "wb") as h:
        pickle.dump((train, val), h)



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
    input_filename = source_test.txt

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





if __name__ == '__main__':
	from seq2seq import Seq2Seq
	m = Seq2Seq(len(DE.vocab), len(EN.vocab))


	m.forward(batch.source, batch.target)


