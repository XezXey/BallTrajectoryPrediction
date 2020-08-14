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
    return pt.nn.Sequential(pt.nn.Linear(in_f, out_f, bias=False))
  else :
    return pt.nn.Sequential(
      pt.nn.Linear(in_f, out_f, bias=True),
      pt.nn.ReLU(),
    )

def create_recurrent_block(in_f, hidden_f, num_layers, is_first_layer=False):
  if is_first_layer:
    return pt.nn.GRU(input_size=in_f, hidden_size=hidden_f, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.)
  else :
    # this need for stacked bidirectional LSTM/GRU/RNN
    return pt.nn.GRU(input_size=in_f*2, hidden_size=hidden_f, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.)

class BiGRUResidualList(pt.nn.Module):
  def __init__(self, input_size, output_size):
    super(BiGRUResidualList, self).__init__()
    # Define the model parameters
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_dim = 32
    self.n_layers = 1
    self.n_stack = 6
    # This will create the Recurrent blocks by specify the input/output features
    self.recurrent_stacked = [self.input_size] + [self.hidden_dim] * self.n_stack
    self.residual_connection = []
    # This will create the FC blocks by specify the input/output features
    self.fc_size = [self.hidden_dim*2, 64, 32, 16, 8, 4, self.output_size]
    # Define the layers
    # LSTM layer with Bi-directional : need to multiply the input size by 2 because there's 2 directional from previous layers
    self.recurrent_blocks = pt.nn.ModuleList([create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=True) if in_f == self.input_size
                                              else create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=False)
                                              for in_f, hidden_f in zip(self.recurrent_stacked, self.recurrent_stacked[1:])])
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

    for idx, recurrent_block in enumerate(self.recurrent_blocks):
      # if idx >= 1:
        # All but not the input layer
        # print("#N forward : {}, SHAPE : {}".format(idx, self.residual_connection[idx-1].shape))
        # print([self.residual_connection[i].shape for i in range(len(self.residual_connection))])

      # Pass the packed sequence to the recurrent blocks 
      out_packed, hidden = recurrent_block(out_packed)

      # Unpacked sequence for residual connection then packed it back for next input
      out_unpacked = pad_packed_sequence(out_packed, batch_first=True, padding_value=-10)[0]
      # Residual in recurrent block
      if idx >= 2:
        # print("[#] RESIDUAL!!!")
        # print(pt.stack((self.residual_connection)).shape)
        # print(pt.sum(pt.stack((self.residual_connection)), dim=0).shape)
        out_unpacked = pt.sum(pt.stack((self.residual_connection)), dim=0)

      # Append the output of previous state
      self.residual_connection.append(out_unpacked.clone())
      # Pack the sequence for next input
      out_packed = pack_padded_sequence(out_unpacked, lengths=lengths, batch_first=True, enforce_sorted=False)

    # Residual from recurrent block to FC
    # print(pt.stack((self.residual_connection)).shape)
    out_unpacked = pt.sum(pt.stack((self.residual_connection)), dim=0)
    out_unpacked = pad_packed_sequence(out_packed, batch_first=True, padding_value=-10)[0]
    # print(out_unpacked.shape)
    # Pass the unpacked(The hidden features from RNN) to the FC layers
    out = self.fc_blocks(out_unpacked)
    # print(out.shape)
    self.residual_connection = []
    return out, (hidden, cell_state)

  def initHidden(self, batch_size):
    hidden = Variable(pt.randn(self.n_layers*2, batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return hidden

  def initCellState(self, batch_size):
    cell_state = Variable(pt.randn(self.n_layers*2, batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return cell_state

