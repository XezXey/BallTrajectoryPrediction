from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt

def create_fc_block(in_f, out_f, is_last_layer=False):
  # Auto create the FC blocks
  if is_last_layer:
    return pt.nn.Sequential(pt.nn.Linear(in_f, out_f, bias=True))
  else :
    return pt.nn.Sequential(
      pt.nn.Linear(in_f, out_f, bias=True),
      pt.nn.ReLU(),
    )

def create_recurrent_block(in_f, hidden_f, num_layers):
  return pt.nn.LSTM(input_size=in_f, hidden_size=hidden_f, num_layers=num_layers, batch_first=True)

class LSTM(pt.nn.Module):
  def __init__(self, input_size, output_size):
    super(LSTM, self).__init__()
    # Define the model parameters
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_dim = 64
    self.n_layers = 8
    self.recurrent_stacked = [self.input_size, 64, 64, self.hidden_dim]
    # This will create the FC blocks by specify the input/output features
    self.fc_size = [self.hidden_dim, 32, 16, 8, self.output_size]
    # Define the layers
    # RNN layer
    self.recurrent_blocks = pt.nn.ModuleList([create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers) for in_f, hidden_f in zip(self.recurrent_stacked, self.recurrent_stacked[1:])])
    # FC
    fc_blocks = [create_fc_block(in_f, out_f, is_last_layer=False) if out_f!=self.output_size
                 else create_fc_block(in_f, out_f, is_last_layer=True)
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
      out_packed, (hidden, cell_state) = recurrent_block(out_packed, (hidden, cell_state))
    # Unpacked the hidden features sequence
    out_unpacked = pad_packed_sequence(out_packed, batch_first=True, padding_value=-1)[0]
    # Pass the unpacked(The hidden features from RNN) to the FC layers
    out = self.fc_blocks(out_unpacked)
    return out, (hidden, cell_state)

  def initHidden(self, batch_size):
    hidden = Variable(pt.randn(self.n_layers, batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return hidden

  def initCellState(self, batch_size):
    cell_state = Variable(pt.randn(self.n_layers, batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return cell_state

