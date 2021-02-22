from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Encoder(pt.nn.Module):
  def __init__(self, input_size, output_size, batch_size, model, residual):
    super(Encoder, self).__init__()
    # Define the model parameters
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.model = model
    self.residual = residual

    # This will create the FC blocks by specify the input/output features
    self.fc_size = [self.input_size, self.input_size, self.input_size, self.output_size]
    # Define the layers
    # FC
    fc_blocks = [self.create_fc_block(in_f, out_f) for in_f, out_f in zip(self.fc_size, self.fc_size[1:])]

    self.fc_blocks = pt.nn.Sequential(*fc_blocks)

  def forward(self, in_f):
    # Pass the Features to FC layers
    out = self.fc_blocks(in_f)
    if self.residual:
      out_skip = out + in_f[..., [0, 1, 2]]
      return out_skip
    return out

  def create_fc_block(self, in_f, out_f):
    # Auto create the FC blocks
    return pt.nn.Sequential(
      pt.nn.Linear(in_f, out_f, bias=True),
      pt.nn.LeakyReLU(negative_slope=0.01),
    )
