#seq to seq

import torch
from namedtensor import ntorch
from torch.distributions import categorical
import torch.nn as nn
import numpy as np


class SyntaxSeq2Seq(nn.Module):

    def __init__(self, in_vocab_size, out_vocab_size, use_pretrained=False, attention=False, embedding_size=128, num_layers=1, dropout=0.2, hidden_size=256): #todo params
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



