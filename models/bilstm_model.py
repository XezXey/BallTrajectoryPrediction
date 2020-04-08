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

class BiLSTM(pt.nn.Module):
  def __init__(self, input_size, output_size):
    super(BiLSTM, self).__init__()
    # Define the model parameters
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_dim = 32
    self.n_layers = 64
    # This will create the FC blocks by specify the input/output features
    self.fc_size = [self.hidden_dim*2, 64, 32, 16, 8, self.output_size]
    # Define the layers
    # LSTM layer with Bi-directional : need to multiply the input size by 2 because there's 2 directional from previous layers
    self.lstm = pt.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=self.n_layers, batch_first=True, bidirectional=True)
    # FC
    fc_blocks = [create_fc_block(in_f, out_f, is_last_layer=False) if out_f!=self.output_size
                 else create_fc_block(in_f, out_f, is_last_layer=True) for in_f, out_f in zip(self.fc_size, self.fc_size[1:])]
    self.fc_blocks = pt.nn.Sequential(*fc_blocks)

  def forward(self, x, hidden, cell_state, lengths):
    # Passing in the input(x) and hidden state into the model and obtaining the outputs
    # Use packed sequence to speed up the RNN/LSTM
    # pack_padded_sequence => RNN => pad_packed_sequence[0] to get the data in batch
    x_packed = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
    out_packed, (hidden, cell_state) = self.lstm(x_packed, (hidden, cell_state))
    out_unpacked = pad_packed_sequence(out_packed, batch_first=True, padding_value=-1)[0]
    # Pass the unpacked(The hidden features from RNN) to the FC layers
    out = self.fc_blocks(out_unpacked)
    return out, (hidden, cell_state)

  def initHidden(self, batch_size):
    hidden = Variable(pt.randn(self.n_layers*2, batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return hidden

  def initCellState(self, batch_size):
    cell_state = Variable(pt.randn(self.n_layers*2, batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return cell_state

