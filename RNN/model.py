import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNClasifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClasifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, seq_lengths):
        # Run all at once (over whole input sequence)
        # Input shape: batch x seq ---transpose---> S x B
        input = input.t()
        batch_size = input.size(1)

        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            hidden = Variable(hidden.cuda())
        else:
            hidden = Variable(hidden)
        # S x B -> S x B x I
        embedded = self.embedding(input)
        gru_input = pack_padded_sequence(embedded, seq_lengths.data.cpu().numpy())

        # To compact weights again call flatten_parameters()
        self.gru.flatten_parameters()
        output, hidden = self.gru(gru_input, hidden)
        
        # Use the last layer output as FC's input
        # No need to unpack since we are going to use hidden
        fc_output = self.fc(hidden[-1])
        return fc_output