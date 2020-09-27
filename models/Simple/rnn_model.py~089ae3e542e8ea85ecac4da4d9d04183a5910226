from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

class RNN(torch.nn.Module):
  def __init__(self, input_size, output_size, hidden_dim, n_layers):
    super(RNN, self).__init__()
    # Define the model parameters
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    # Define the layers
    # RNN layer
    self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
    # FC
    self.fc1 = torch.nn.Linear(hidden_dim, 16, bias=True)
    self.fc2 = torch.nn.Linear(16, 4, bias=True)
    self.fc3 = torch.nn.Linear(4, output_size, bias=True)
    self.relu = torch.nn.ReLU()

  def forward(self, x, hidden):
    # Passing in the input(x) and hidden state into the model and obtaining the outputs
    out, hidden = self.rnn(x, hidden)
    # Reshaping the output such that it can be fit into the fc layer
    out = out.contiguous().view(-1, self.hidden_dim) # Flatten
    out = self.fc1(out)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.relu(out)
    out = self.fc3(out)
    return out, hidden

  def initHidden(self, batch_size):
    hidden = Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim, dtype=torch.float32)).cuda()
    return hidden

