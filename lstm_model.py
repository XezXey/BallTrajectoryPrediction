from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

class LSTM(torch.nn.Module):
  def __init__(self, input_size, output_size, hidden_dim, n_layers):
    super(LSTM, self).__init__()
    # Define the model parameters
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    # Define the layers
    # RNN layer
    self.lstm1 = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=False)
    # FC
    self.fc1 = torch.nn.Linear(hidden_dim, 32, bias=True)
    self.fc2 = torch.nn.Linear(32, 16, bias=True)
    self.fc3 = torch.nn.Linear(16, 8, bias=True)
    self.fc4 = torch.nn.Linear(8, output_size, bias=True)
    self.relu1 = torch.nn.ReLU()
    self.relu2 = torch.nn.ReLU()
    self.relu3 = torch.nn.ReLU()

  def forward(self, x, hidden, cell_state):
    # Passing in the input(x) and hidden state into the model and obtaining the outputs
    out, (hidden, cell_state) = self.lstm1(x, (hidden, cell_state))
    # Reshaping the output such that it can be fit into the fc layer
    out = out.contiguous().view(-1, self.hidden_dim) # Flatten
    out = self.fc1(out)
    out = self.relu1(out)
    out = self.fc2(out)
    out = self.relu2(out)
    out = self.fc3(out)
    out = self.relu3(out)
    out = self.fc4(out)
    return out, (hidden, cell_state)

  def initHidden(self, batch_size):
    hidden = Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim, dtype=torch.float32)).cuda()
    return hidden

  def initCellState(self, batch_size):
    cell_state = Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim, dtype=torch.float32)).cuda()
    return cell_state

