# Torch
import torch
# Text text processing library and methods for pretrained word embeddings
from torchtext import data, datasets
# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
import spacy

from seq2seq import Seq2Seq
#from syntaxseq2seq import SyntaxSeq2Seq
import time

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
device = torch.device('cuda:0')
train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=BATCH_SIZE, device=device,
                                                  repeat=False, sort_key=lambda x: len(x.src))

#batch = next(iter(train_iter))


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

import torch
from namedtensor import ntorch
from torch.distributions import categorical
import torch.nn as nn
import numpy as np
device = torch.device('cuda:0')

class SyntaxSeq2Seq(nn.Module):

    def __init__(self, in_vocab_size, out_vocab_size, use_pretrained=False, attention=False, embedding_size=128, num_layers=2, dropout=0.2, hidden_size=256): #todo params
        super(SyntaxSeq2Seq, self).__init__()
        self.attention = attention
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.in_embedding = ntorch.nn.Embedding(
                    in_vocab_size, embedding_size
                )
        self.out_embedding = ntorch.nn.Embedding(
                    out_vocab_size, embedding_size
                )

        if use_pretrained:
            self.in_embedding.weight.data.copy_(in_pretrained_embeddings)
            self.in_embedding.weight.requires_grad = mode == "nonstatic"

            self.out_embedding.weight.data.copy_(out_pretrained_embeddings)
            self.out_embedding.weight.requires_grad = mode == "nonstatic"

        self.encoder = ntorch.nn.LSTM(embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False,
                            dropout=dropout,
                            bidirectional=True
                            ).spec("embedding", "srcSeqlen", "rnnOutput")

        self.decoder = ntorch.nn.LSTM(embedding_size,
                            hidden_size=2*hidden_size,
                            num_layers=num_layers,
                            batch_first=False,
                            dropout=dropout
                            ).spec("embedding", "trgSeqlen", "rnnOutput")

        self.syntax_decoder = ntorch.nn.LSTM(embedding_size,
                            hidden_size=2*hidden_size,
                            num_layers=num_layers,
                            batch_first=False,
                            dropout=dropout
                            ).spec("embedding", "trgSeqlen", "rnnOutput")
        
        if self.attention:
            self.fc = ntorch.nn.Linear(
                4 * hidden_size, out_vocab_size
            ).spec("rnnOutput", "outVocab")
        else:  
            self.fc = ntorch.nn.Linear(
                2 * hidden_size, out_vocab_size
            ).spec("rnnOutput", "outVocab")

        self.syntax_fc = ntorch.nn.Linear(
                2 * hidden_size, out_vocab_size
            ).spec("rnnOutput", "outVocab")

        self.lossfn = ntorch.nn.CrossEntropyLoss(reduction="none").spec("outVocab") 

    def forward(self, source, target=None, teacher_forcing=1., max_length=20, encode_only=False):
        if target:
          max_length = target.shape["trgSeqlen"]
        x = self.in_embedding(source)
        out, (h, c) = self.encoder(x)
        h = ntorch.cat((h[{"layers": slice(0,1)}], h[{"layers": slice(1,2)}]), dim="rnnOutput")
        c = ntorch.cat((c[{"layers": slice(0,1)}], c[{"layers": slice(1,2)}]), dim="rnnOutput")

        if self.attention:
            def attend(x_t):
                alpha = out.dot("rnnOutput", x_t).softmax("srcSeqlen")
                context = alpha.dot("srcSeqlen", out)
                return context
        
        batch_size = source.shape["batch"]
        output_dists = ntorch.zeros((batch_size, max_length, self.out_vocab_size), names=("batch", "trgSeqlen", "outVocab"), device=device)
        output_seq = ntorch.zeros((batch_size, max_length), names=("batch", "trgSeqlen"), device=device)
        #for the above, should set zeroith index to SOS        
        
        score = ntorch.zeros((batch_size, max_length), names=("batch", "trgSeqlen"), device=device)
        
        if encode_only:
          return score, out, (h, c), output_seq
        
        for t in range(max_length - 1): #Oh god
            if t==0:
                # always start with SOS token
                next_input = ntorch.ones((batch_size, 1), names=("batch", "trgSeqlen"), device=device).long()
                next_input *= EN.vocab.stoi["<s>"]
            elif np.random.random() < teacher_forcing and target:  # we will force
                next_input = target[{"trgSeqlen": slice(t, t+1)}]
            else:
                next_input = sample
                
                
            x_t, (h, c) = self.decoder(self.out_embedding(next_input), (h, c))

            if t == 0:
                syntax_out, (s_h, s_c) = self.syntax_decoder(self.out_embedding(next_input))
            else:
                syntax_out, (s_h, s_c) = self.syntax_decoder(self.out_embedding(next_input), (s_h, s_c))

            if self.attention:
                fc = self.fc(ntorch.cat([attend(x_t), x_t], dim="rnnOutput") )
            else:
                fc = self.fc(x_t)
              
            s_fc = self.syntax_fc(syntax_out).sum("trgSeqlen")
            s_fc = s_fc.log_softmax("outVocab")

            dist = ntorch.distributions.Categorical(logits=fc, dim_logit="outVocab")
            sample = dist.sample()
            
            fc = fc.sum("trgSeqlen")

            next_token = (sample) if not target else target[{"trgSeqlen": slice(t+1, t+2)}]#TODO
            
            #this is the line where the syntax thing does it's stuff
            fc = fc.log_softmax("outVocab") + s_fc

            indices = next_token.sum("trgSeqlen").rename("batch", "indices")
            batch_indices = ntorch.tensor(torch.tensor(np.arange(fc.shape["batch"]), device=device), ("batchIndices"))

            newsc = fc.index_select("outVocab", indices).index_select("indices", batch_indices).get("batchIndices", 0)
            
            score[{"trgSeqlen": t+1}] = newsc 

            output_seq[{"trgSeqlen":t+1}] = next_token.sum("trgSeqlen") #todo 
            output_dists[{"trgSeqlen":t+1}] = fc #Todo
 
        return output_seq, output_dists, score


    def beam_decode(self, inputs, beam_size=100, max_length=4):
        
        #encode inputs
        score, out, (h, c), output_seq = self(inputs, max_length=max_length, encode_only=True)
        
        enc_out = out
        
        state = (out, (h,c))
        beam = [(output_seq, score, state, None)]

        
        #run first step
       
        for l in range(max_length-1):
            new_beam = []
            for output_seq, score, state, syntax_state in beam:
                new_beam.extend(self.all_possible_from_one_step(l, output_seq, score, state, syntax_state, enc_out))
            
            new_beam = sorted(new_beam, key=lambda entry: -entry[1]) # i think this is right....
            beam = new_beam[:beam_size]

        return beam
            
        
    def all_possible_from_one_step(self, l, output_seq, score, state, syntax_state, enc_out):
        if not self.active(output_seq):
            return [(output_seq, score, state)]
        else: 
            if l == 0:
                output_seq[{"trgSeqlen":0}] == EN.vocab.stoi["<s>"]
            
        return self.decode_one_step(l, output_seq, score, state, syntax_state, enc_out)
        
    def decode_one_step(self, t, output_seq, score, state, syntax_state, enc_out):
        
        if self.attention:
            def attend(x_t):
                alpha = enc_out.dot("rnnOutput", x_t).softmax("srcSeqlen")
                context = alpha.dot("srcSeqlen", enc_out)
                return context
        
        
        h, c = state[-1]
        if syntax_state: (s_h, s_c) = syntax_state[-1]

        next_input = output_seq[{"trgSeqlen": slice(t, t+1)}].long()

        x_t, (h, c) = self.decoder(self.out_embedding(next_input), (h, c))

        if syntax_state == None:
            syntax_out, (s_h, s_c) = self.syntax_decoder(self.out_embedding(next_input))
        else:
            syntax_out, (s_h, s_c) = self.syntax_decoder(self.out_embedding(next_input), (s_h, s_c))

        if self.attention:
            fc = self.fc(ntorch.cat([attend(x_t), x_t], dim="rnnOutput") )
        else:
            fc = self.fc(x_t)
          
        s_fc = self.syntax_fc(syntax_out).sum("trgSeqlen")
        s_fc = s_fc.log_softmax("outVocab")

        fc = fc.sum("trgSeqlen")

        fc = fc.log_softmax("outVocab") + s_fc

        state = x_t, (h, c)
        
        syntax_state = syntax_out, (s_h, s_c)
        #can instead use argmax ... 
        #next_tokens = fc.argmax("")
        #ntorch.tensor(topk, names=dim_names)
        #max, argmax = fc.topk("dim2", k)
    
        k = 100
        _, argmax = fc.topk("outVocab", k)
        #print("argmax", argmax)
        lst = []
        for i in range(k): #TODO fix this line or whatever
            import copy
            output_seq = copy.deepcopy(output_seq)
            
            output_seq[{"trgSeqlen":t+1}] = argmax[{"outVocab":i}] #TODO fix this line or whatever

            next_token = output_seq[{"trgSeqlen": slice(t+1, t+2)}].long()

            indices = next_token.sum("trgSeqlen").rename("batch", "indices")
            batch_indices = ntorch.tensor(torch.tensor(np.arange(fc.shape["batch"]), device=device), ("batchIndices"))
            newsc = fc.index_select("outVocab", indices).index_select("indices", batch_indices).get("batchIndices", 0)
            score[{"trgSeqlen": t+1}] = newsc

            assert output_seq[{"trgSeqlen":t+1}].long() == next_token.sum("trgSeqlen") #todo 
            #output_dists[{"trgSeqlen":t+1}] = fc
            
            lst.append((output_seq, score, state, syntax_state))
        return lst

        
    def active(self, output_seq):
        #print(output_seq)
        #print(output_seq[output_seq != 0])
        #print(output_seq[output_seq != 0].nelement())
        if (output_seq != 0).sum("batch") == 0: return True
        if (output_seq[output_seq != 0].get("trgSeqlen", -1) == 1).sum("batch") == 1: return False
        return True
      

    def loss(self, source, target):
        _, output_dists, score = self(source, target)
        unmasked_loss = self.lossfn(output_dists, target)
        mask = (target != 1.).float()
        total_length = mask.sum("trgSeqlen")
        masked_loss = mask * unmasked_loss
        total_loss = masked_loss.sum("trgSeqlen")
        total_loss = total_loss.div(total_length)
        
        return ntorch.exp(total_loss.mean())

def train(model, debug=False):
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

                import dill
                with open("model.p", 'rb') as h:
                    dill.dump(model, h)
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
	#from seq2seq import Seq2Seq
	model = SyntaxSeq2Seq(len(DE.vocab), len(EN.vocab), attention=True, dropout=0.5)

	train(model)



