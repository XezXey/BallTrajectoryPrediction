from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt

class BiGRU(pt.nn.Module):
  def __init__(self, input_size, output_size, batch_size, model):
    super(BiGRU, self).__init__()
    # Define the model parameters
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_dim = 32
    self.n_layers = 1
    self.batch_size = batch_size
    self.n_stack = 0
    self.model = model
    # This will create the Recurrent blocks by specify the input/output features
    self.recurrent_stacked = [self.input_size, self.hidden_dim, self.hidden_dim, self.hidden_dim]
    # This will create the FC blocks by specify the input/output features
    self.fc_size = [self.hidden_dim*2, 32, 16, 8, 4, self.output_size]
    # Define the layers
    # GRU layer with Bi-directional : need to multiply the input size by 2 because there's 2 directional from previous layers
    self.recurrent_blocks = pt.nn.ModuleList([self.create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=True) if in_f == self.input_size
                                              else self.create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=False)
                                              for in_f, hidden_f in zip(self.recurrent_stacked, self.recurrent_stacked[1:])])
    # FC
    fc_blocks = [self.create_fc_block(in_f, out_f, is_last_layer=False) if out_f!=self.output_size
                 else self.create_fc_block(in_f, out_f, is_last_layer=True)
                 for in_f, out_f in zip(self.fc_size, self.fc_size[1:])]

    self.fc_blocks = pt.nn.Sequential(*fc_blocks)

  def forward(self, x, hidden, cell_state, lengths):
    # Passing in the input(x) and hidden state into the model and obtaining the outputs
    # Use packed sequence to speed up the RNN/LSTM
    # pack_padded_sequence => RNN => pad_packed_sequence[0] to get the data in batch
    x_packed = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
    out_packed = x_packed

    for recurrent_block in self.recurrent_blocks:
      # Pass the packed sequence to the recurrent blocks 
      out_packed, hidden = recurrent_block(out_packed, self.initHidden(batch_size=self.batch_size))
    out_unpacked = pad_packed_sequence(out_packed, batch_first=True, padding_value=-10)[0]
    # Pass the unpacked(The hidden features from RNN) to the FC layers
    out = self.fc_blocks(out_unpacked)
    return out, (hidden, cell_state)

  def create_fc_block(self, in_f, out_f, is_last_layer=False):
    # Auto create the FC blocks
    if is_last_layer:
      if self.model=='flag':
        return pt.nn.Sequential(
          pt.nn.Linear(in_f, out_f, bias=True),
          pt.nn.Sigmoid()
        )
      else:
        return pt.nn.Sequential(
          pt.nn.Linear(in_f, out_f, bias=True),)
    else :
      return pt.nn.Sequential(
        pt.nn.Linear(in_f, out_f, bias=True),
        pt.nn.LeakyReLU(negative_slope=0.01),
      )

  def create_recurrent_block(self, in_f, hidden_f, num_layers, is_first_layer=False):
    if is_first_layer:
      return pt.nn.GRU(input_size=in_f, hidden_size=hidden_f, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.)
    else :
      # this need for stacked bidirectional LSTM/GRU/RNN
      return pt.nn.GRU(input_size=in_f*2, hidden_size=hidden_f, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.)

  def initHidden(self, batch_size):
    hidden = Variable(pt.zeros(self.n_layers*2, batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return hidden

  def initCellState(self, batch_size):
    cell_state = Variable(pt.zeros(self.n_layers*2, batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return cell_state

