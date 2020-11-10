from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt

class BiLSTMResidualTrainableInit(pt.nn.Module):
  def __init__(self, input_size, output_size, batch_size, bidirectional, trainable_init, model):
    super(BiLSTMResidualTrainableInit, self).__init__()
    # Define the model parameters
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.hidden_dim = 32
    self.n_layers = 1
    self.n_stack = 4
    self.model = model
    self.bidirectional_flag = bidirectional
    if bidirectional:
      self.bidirectional = 2
    else:
      self.bidirectional = 1

    # For a  initial state
    self.trainable_init = trainable_init
    if not self.trainable_init:
      self.h, self.c = self.initial_state()
    else:
      self.h, self.c = self.initial_learnable_state()
      self.register_parameter('h', self.h)
      self.register_parameter('c', self.c)

    # This will create the Recurrent blocks by specify the input/output features
    self.recurrent_stacked = [self.input_size] + [self.hidden_dim] * self.n_stack
    # This will create the FC blocks by specify the input/output features
    self.fc_size = [self.hidden_dim*self.bidirectional, 32, 16, 8, 4, self.output_size]
    # Define the layers
    # LSTM layer with Bi-directional : need to multiply the input size by 2 because there's 2 directional from previous layers
    self.recurrent_blocks = pt.nn.ModuleList([self.create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=True) if in_f == self.input_size
                                              else self.create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=False)
                                              for in_f, hidden_f in zip(self.recurrent_stacked, self.recurrent_stacked[1:])])
    # FC
    fc_blocks = [self.create_fc_block(in_f, out_f, is_last_layer=False) if out_f!=self.output_size
                 else self.create_fc_block(in_f, out_f, is_last_layer=True)
                 for in_f, out_f in zip(self.fc_size, self.fc_size[1:])]

    self.fc_blocks = pt.nn.Sequential(*fc_blocks)

  def forward(self, in_f, lengths, hidden=None, cell_state=None):
    # Passing in the input(x) and hidden state into the model and obtaining the outputs
    # Use packed sequence to speed up the RNN/LSTM
    # pack_padded_sequence => RNN => pad_packed_sequence[0] to get the data in batch
    in_f_packed = pack_padded_sequence(in_f, lengths=lengths, batch_first=True, enforce_sorted=False)
    out_packed = in_f_packed
    residual = pt.Tensor([0.]).cuda()
    for idx, recurrent_block in enumerate(self.recurrent_blocks):
      # print("IDX = {}".format(idx), self.h[idx], self.c[idx])
      # Pass the packed sequence to the recurrent blocks with the skip connection
      if self.trainable_init:
        init_h = self.h.repeat(1, 1, self.batch_size, 1)[idx]
        init_c = self.c.repeat(1, 1, self.batch_size, 1)[idx]
      else:
        init_h, init_c = self.initial_state()
      if idx == 0:
        # Only first time that no skip connection from input to other networks
        out_packed, (hidden, cell_state) = recurrent_block(out_packed, (init_h, init_c))
        residual = self.get_residual(out_packed=out_packed, lengths=lengths, residual=residual, apply_skip=False)
      else:
        out_packed, (hidden, cell_state) = recurrent_block(residual, (init_h, init_c))
        residual = self.get_residual(out_packed=out_packed, lengths=lengths, residual=residual, apply_skip=True)

    # Residual from recurrent block to FC
    residual = pad_packed_sequence(residual, batch_first=True, padding_value=-10)[0]
    # Pass the unpacked(The hidden features from RNN) to the FC layers
    out = self.fc_blocks(residual)
    return out, (hidden, cell_state)

  def get_residual(self, out_packed, lengths, residual, apply_skip):
    # Unpacked sequence for residual connection then packed it back for next input
    out_unpacked = pad_packed_sequence(out_packed, batch_first=True, padding_value=-10)[0]

    if apply_skip:
      residual = pad_packed_sequence(residual, batch_first=True, padding_value=-10)[0]
      residual = residual + out_unpacked
    else:
      residual = out_unpacked
    # Pack the sequence for next input
    residual = pack_padded_sequence(residual, lengths=lengths, batch_first=True, enforce_sorted=False)
    return residual

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
      return pt.nn.LSTM(input_size=in_f, hidden_size=hidden_f, num_layers=num_layers, batch_first=True, bidirectional=self.bidirectional_flag, dropout=0.)
    else :
      # this need for stacked bidirectional LSTM/LSTM/RNN
      return pt.nn.LSTM(input_size=in_f*self.bidirectional, hidden_size=hidden_f, num_layers=num_layers, batch_first=True, bidirectional=self.bidirectional_flag, dropout=0.)

  def initial_state(self):
    h = Variable(pt.zeros(self.n_layers*self.bidirectional, self.batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    c = Variable(pt.zeros(self.n_layers*self.bidirectional, self.batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return h, c

  def initial_learnable_state(self):
    # Initial the hidden/cell state as model parameters
    # 1 refer the batch size which is we need to copy the initial state to every sequence in a batch
    h = pt.nn.Parameter(pt.randn(self.n_stack, self.n_layers*self.bidirectional, 1, self.hidden_dim, dtype=pt.float32).cuda(), requires_grad=self.trainable_init).cuda()
    c = pt.nn.Parameter(pt.randn(self.n_stack, self.n_layers*self.bidirectional, 1, self.hidden_dim, dtype=pt.float32).cuda(), requires_grad=self.trainable_init).cuda()
    return h, c

