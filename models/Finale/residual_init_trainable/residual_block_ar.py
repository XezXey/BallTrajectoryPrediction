from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt
import tqdm

class ResNetLayer_AR(pt.nn.Module):
  def __init__(self, input_size, output_size, batch_size, bidirectional, trainable_init, model, autoregressive=False, n_stack=4, hidden_dim=32, n_block=3):
    super().__init__()
    # Define the model parameters
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.hidden_dim = hidden_dim
    self.n_stack = n_stack
    self.n_layers = 1
    self.model = model
    self.bidirectional_flag = bidirectional
    self.autoregressive = autoregressive
    if bidirectional:
      self.bidirectional = 2
    else:
      self.bidirectional = 1
    ############################## Hidden dimension ############################
    # This will create the Recurrent blocks by specify the input/output features
    self.recurrent_stacked = [self.input_size] + [self.hidden_dim] * self.n_stack
    # This will create the FC blocks by specify the input/output features
    self.fc_size = [self.hidden_dim*self.bidirectional, 32, 16, 8, 4, self.input_size]

    # Create Resnet Block
    self.blocks = pt.nn.Sequential(*[ResidualBlock_AR(input_size=input_size, output_size=output_size, batch_size=batch_size, bidirectional=bidirectional,
                                                   trainable_init=trainable_init, model=model, n_stack=n_stack, hidden_dim=hidden_dim, autoregressive=self.autoregressive) for _ in range(n_block)])
    self.fc_out = pt.nn.Linear(input_size, output_size)

    # Manually set weight for residual
    # self.manually_set_weight()

  def forward(self, in_f, lengths, hidden=None, cell_state=None):
    x = {'in_f':in_f, 'lengths':lengths, 'hidden':hidden, 'cell_state':cell_state}
    residual = self.blocks(x)
    if model == 'refinement':
      x = residual['in_f']
    else:
      x = self.fc_out(residual['in_f'])

    return x, (residual['hidden'], residual['cell_state'])

  def set_init(self):
    for block in self.blocks:
      block.set_init()

  def manually_set_weight(self):
    for block in self.blocks:
      for name, param in block.named_parameters():
        if 'bias' in name:
          pt.nn.init.constant(param, 0.0)
        elif 'weight' in name:
          pt.nn.init.constant(param, 0.0)



class ResidualBlock_AR(pt.nn.Module):
  def __init__(self, input_size, output_size, batch_size, bidirectional, trainable_init, model, autoregressive=False, n_stack=4, hidden_dim=32):
    super(ResidualBlock_AR, self).__init__()
    # Define the model parameters
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.hidden_dim = hidden_dim
    self.n_stack = n_stack
    self.n_layers = 1
    self.model = model
    self.bidirectional_flag = bidirectional
    self.autoregressive = autoregressive
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

    # Saving current state for autoregressive
    self.current_h = self.h.clone()
    self.current_c = self.c.clone()

    ############################## Hidden dimension ############################
    # This will create the Recurrent blocks by specify the input/output features
    self.recurrent_stacked = [self.input_size] + [self.hidden_dim] * self.n_stack
    # This will create the FC blocks by specify the input/output features
    self.fc_size = [self.hidden_dim*self.bidirectional, 32, 16, 8, 4, self.input_size]

    ################################# Layer ####################################
    # LSTM layer with Bi-directional : need to multiply the input size by 2 because there's 2 directional from previous layers
    recurrent_blocks = [self.create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=True) if in_f == self.input_size
                                              else self.create_recurrent_block(in_f=in_f, hidden_f=hidden_f, num_layers=self.n_layers, is_first_layer=False)
                                              for in_f, hidden_f in zip(self.recurrent_stacked, self.recurrent_stacked[1:])]
    self.recurrent_blocks = pt.nn.Sequential(*recurrent_blocks)

    # FC
    fc_blocks = [self.create_fc_block(in_f, out_f, is_last_layer=False) if out_f!=self.input_size
                 else self.create_fc_block(in_f, out_f, is_last_layer=True)
                 for in_f, out_f in zip(self.fc_size, self.fc_size[1:])]

    self.fc_blocks = pt.nn.Sequential(*fc_blocks)

    # Activation
    self.relu = pt.nn.LeakyReLU(negative_slope=0.01)

  def forward(self, x):
    in_f = x['in_f']
    lengths = x['lengths']
    hidden = x['hidden']
    cell_state = x['cell_state']
    if self.autoregressive:
      return self.auto_regressive(in_f=in_f, lengths=lengths)
    else:
      return self.forward_pass(in_f=in_f, lengths=lengths)

  def set_init(self):
    self.current_h = self.h.clone()
    self.current_c = self.c.clone()

  def auto_regressive(self, in_f, lengths, hidden=None, cell_state=None):
    batch_size = in_f.shape[0]
    # Passing in the input(x) and hidden state into the model and obtaining the outputs
    out_f = pt.unsqueeze(in_f[[0], [0], ...], dim=0)
    for block_idx, recurrent_block in enumerate(self.recurrent_blocks):
      # Pass the packed sequence to the recurrent blocks with the skip connection
      # print("INIT : ", init_h)
      out_f, (hidden, cell_state) = recurrent_block(out_f, (self.current_h[block_idx], self.current_c[block_idx]))

      # Update hidden/cell state for next sequence
      self.current_h[block_idx] = hidden
      self.current_c[block_idx] = cell_state

    # print("MOD H : ", self.current_h[block_idx][0])
    # print("INIT : ", self.h[block_idx][0])

    # Residual from recurrent block to FC
    # Pass the unpacked(The hidden features from RNN) to the FC layers
    # print("OUTPUT LSTM : ", out)
    # print(out_f.shape)
    out = self.fc_blocks(out_f)
    # print(out.shape)
    # print("OUTPUT FC: ", out)
    # print(out_block.shape)
    # print("OUTPUT BLOCK: ", out_block.shape)
    # print("OUTPUT BLOCK: ", out_block)
    # exit()
    return {'in_f':out_block, 'lengths':lengths, 'hidden':hidden, 'cell_state':cell_state}


  def forward_pass(self, in_f, lengths, hidden=None, cell_state=None):
    # Passing in the input(x) and hidden state into the model and obtaining the outputs
    # Use packed sequence to speed up the RNN/LSTM
    # pack_padded_sequence => RNN => pad_packed_sequence[0] to get the data in batch
    # print("INPUT RESIDUAL : ", in_f)
    in_f_packed = pack_padded_sequence(in_f, lengths=lengths, batch_first=True, enforce_sorted=False)
    out_packed = in_f_packed
    for idx, recurrent_block in enumerate(self.recurrent_blocks):
      # print("IDX = {}".format(idx), self.h[idx], self.c[idx])
      # Pass the packed sequence to the recurrent blocks with the skip connection
      if self.trainable_init:
        init_h = self.h.repeat(1, 1, self.batch_size, 1)[idx]
        init_c = self.c.repeat(1, 1, self.batch_size, 1)[idx]
      else:
        init_h, init_c = self.initial_state()

      out_packed, (hidden, cell_state) = recurrent_block(out_packed, (init_h, init_c))

    # Residual from recurrent block to FC
    out = pad_packed_sequence(out_packed, batch_first=True, padding_value=-10)[0]
    # Pass the unpacked(The hidden features from RNN) to the FC layers
    out = self.fc_blocks(out)
    # print("OUTPUT RESIDUAL : ", out)
    out_block = self.relu(out+in_f)
    # print("OUTPUT APPLY RESIDUAL : ", out+in_f)
    # print("OUTPUT APPLY RELU : ", out_block)
    # exit()
    return {'in_f':out_block, 'lengths':lengths, 'hidden':hidden, 'cell_state':cell_state}

  def create_fc_block(self, in_f, out_f, is_last_layer=False):
    # Auto create the FC blocks
    if is_last_layer:
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
    h = pt.nn.Parameter(pt.zeros(self.n_stack, self.n_layers*self.bidirectional, 1, self.hidden_dim, dtype=pt.float32).cuda(), requires_grad=self.trainable_init).cuda()
    c = pt.nn.Parameter(pt.zeros(self.n_stack, self.n_layers*self.bidirectional, 1, self.hidden_dim, dtype=pt.float32).cuda(), requires_grad=self.trainable_init).cuda()
    return h, c

