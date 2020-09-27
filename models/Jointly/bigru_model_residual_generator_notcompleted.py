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
      pt.nn.LeakyReLU(negative_slope=0.2),
    )

def create_recurrent_block(in_f, hidden_f, num_layers, is_first_layer=False):
  if is_first_layer:
    return pt.nn.GRU(input_size=in_f, hidden_size=hidden_f, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.)
  else :
    # this need for stacked bidirectional LSTM/GRU/RNN
    return pt.nn.GRU(input_size=in_f*2, hidden_size=hidden_f, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.)

class Generator(pt.nn.Module):
  def __init__(self, input_size, output_size):
    super(Generator, self).__init__()
    # Define the model parameters
    self.input_size = input_size
    self.input_size_eot = 2
    self.input_size_depth = 3
    self.output_size_eot = 1
    self.output_size_depth = 1
    self.hidden_dim_eot = 32
    self.hidden_dim_depth = 32
    self.n_layers_eot = 1
    self.n_layers_depth = 1
    self.n_stack_eot = 4
    self.n_stack_depth = 4
    # This will create the Recurrent blocks by specify the input/output features for EOT and Depth
    self.recurrent_stacked_eot = [self.input_size_eot] + [self.hidden_dim_eot] * self.n_stack_eot
    self.recurrent_stacked_depth = [self.input_size_depth] + [self.hidden_dim_depth] * self.n_stack_depth
    # This will create the FC blocks by specify the input/output features for EOT and Depth
    self.fc_size_eot = [self.hidden_dim_eot*2, 32, 16, 8, 4, self.output_size_eot]
    self.fc_size_depth = [self.hidden_dim_depth*2, 32, 16, 8, 4, self.output_size_depth]
    # Define the layers
    # With Bi-directional : need to multiply the input size by 2 because there's 2 directional from previous layers
    self.recurrent_blocks_eot = pt.nn.ModuleList([create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=True) if in_f == self.input_size_eot
                                              else create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=False)
                                              for in_f, hidden_f in zip(self.recurrent_stacked_eot, self.recurrent_stacked_eot[1:])])

    self.recurrent_blocks_depth = pt.nn.ModuleList([create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=True) if in_f == self.input_size_depth
                                              else create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=False)
                                              for in_f, hidden_f in zip(self.recurrent_stacked_depth, self.recurrent_stacked_depth[1:])])
    # FC
    fc_blocks_eot = [create_fc_block(in_f, out_f, is_last_layer=False) if out_f!=self.output_size_eot
                       else create_fc_block(in_f, out_f, is_last_layer=True)
                       for in_f, out_f in zip(self.fc_size_eot, self.fc_size_eot[1:])]
    fc_blocks_depth = [create_fc_block(in_f, out_f, is_last_layer=False) if out_f!=self.output_size_depth
                       else create_fc_block(in_f, out_f, is_last_layer=True)
                       for in_f, out_f in zip(self.fc_size_depth, self.fc_size_depth[1:])]

    self.fc_blocks_eot = pt.nn.Sequential(*fc_blocks_eot)
    self.fc_blocks_depth = pt.nn.Sequential(*fc_blocks_depth)

  def forward(self, x, hidden, cell_state, lengths):
    # Passing in the input(x) and hidden state into the model and obtaining the outputs
    # Use packed sequence to speed up the RNN/LSTM
    # pack_padded_sequence => RNN => pad_packed_sequence[0] to get the data in batch
    x_packed = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
    out_packed = x_packed
    residual_eot = pt.Tensor([0.]).cuda()
    residual_depth = pt.Tensor([0.]).cuda()

    for idx, recurrent_block in enumerate(self.recurrent_blocks):
      # Pass the packed sequence to the recurrent blocks with the skip connection
      if idx == 0:
        # Only first time that no skip connection from input to other networks
        out_packed, hidden = recurrent_block(out_packed)
        residual = self.get_residual(out_packed=out_packed, lengths=lengths, residual=residual, apply_skip=False)
      else:
        out_packed, hidden = recurrent_block(residual)
        residual = self.get_residual(out_packed=out_packed, lengths=lengths, residual=residual, apply_skip=True)

    # Residual from recurrent block to FC
    residual = pad_packed_sequence(residual, batch_first=True, padding_value=-10)[0]
    # Pass the unpacked(The hidden features from RNN) to the FC layers
    out = self.fc_blocks(residual)
    return out, (hidden, cell_state)

  def initHidden(self, batch_size):
    hidden = Variable(pt.randn(self.n_layers*2, batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return hidden

  def initCellState(self, batch_size):
    cell_state = Variable(pt.randn(self.n_layers*2, batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return cell_state

  def get_residual(self, out_packed, lengths, residual, apply_skip):
    # Unpacked sequence for residual connection then packed it back for next input
    out_unpacked = pad_packed_sequence(out_packed, batch_first=True, padding_value=-10)[0]

    if apply_skip:
      residual = pad_packed_sequence(residual, batch_first=True, padding_value=-10)[0]
      residual += out_unpacked
    else:
      residual = out_unpacked
    # Pack the sequence for next input
    residual = pack_padded_sequence(residual, lengths=lengths, batch_first=True, enforce_sorted=False)
    return residual

