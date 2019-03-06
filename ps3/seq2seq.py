#seq to seq

import torch
import namedtensor as ntorch
from ntorch.distributions import Categorical


class Seq2Seq(nn.Module):

    def __init__(self, in_vocab_size, out_vocab_size, embedding_size=128, num_layers=1, dropout=0.2, hidden_size=512): #todo params
        super(seq2seq, self).__init__()

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
                            dropout=dropout
                            ).spec("embedding", "in_seqlen", "rnn_output")

        self.decoder = ntorch.nn.LSTM(embedding_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=False,
                            dropout=dropout
                            ).spec("embedding", "out_seqlen", "rnn_output")
        
        self.fc = ntorch.nn.Linear(
            hidden_size, vocab_size
        ).spec("rnn_output", "out_vocab")

        self.lossfn = ntorch.nn.CrossEntropyLoss().spec("out_vocab") 

    def forward(self, source, target=None, teacher_forcing=0., max_length = 50):
        x = self.in_embedding(source)
        out, (h, c) = self.encoder(x)
        
        output_dists = ntorch.zeros((batch_size, max_length, out_vocab_size), names=("batch", "out_seqlen", "out_vocab"))
        output_seq = ntorch.zeros((batch_size, max_length), names=("batch", "out_seqlen"))
        #for the above, should set zeroith index to SOS

        score = ntorch.zeros((batch_size), names=("batch"))
        for t in range(max_length): #Oh god
            if np.rand() < teacher_forcing or t==0:  # we will force
                if target: next_input = target.get("out_seqlen", t)
                else: next_input = (ntorch.ones((batch_size,), names="batch")*EN.vocab.stoi["<s>"]).longtensor() #TODO
            else:
                next_input = sample # TODO
            x_t, (h, c) = self.decoder(self.out_embedding(next_input), (h, c))

            assert x_t.shape["out_seqlen"] == 1 #idk if this makes sense

            dist = Categorical(self.fc(x_t), "out_vocab")
            sample = dist.sample()


            next_token = (sample) if not target else target.get("out_seqlen", t+1)#TODO
            score += torch.log_softmax(dist).get("out_vocab", next_token)
            output_seq[{"out_seqlen":t+1}] = next_token #todo 
            output_dists[{"out_seqlen":t+1}] = dist #Todo
 
        return output_seq, output_dists, score


    def beam_decode(self, beam_size=100, max_length=3):


    def loss(self, source, target):


        _, _, score = self(source, target)
        return score.mean() #self.lossfn(output_dists, target) # TODO



