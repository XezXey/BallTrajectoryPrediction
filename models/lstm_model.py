from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch as pt
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import matplotlib.pyplot as plt

class LSTM(pt.nn.Module):
  def __init__(self, input_size, output_size, hidden_dim, n_layers):
    super(LSTM, self).__init__()
    # Define the model parameters
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    # Define the layers
    # RNN layer
    self.lstm1 = pt.nn.LSTM(input_size=input_size, hidden_size=self.hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=False, dropout=0.5)
    self.lstm2 = pt.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=False, dropout=0.5)
    self.lstm3 = pt.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=False, dropout=0.5)
    # FC
    self.fc1 = pt.nn.Linear(self.hidden_dim, 32, bias=True)
    self.fc2 = pt.nn.Linear(32, 16, bias=True)
    self.fc3 = pt.nn.Linear(16, 8, bias=True)
    self.fc4 = pt.nn.Linear(8, output_size, bias=True)
    self.relu1 = pt.nn.ReLU()
    self.relu2 = pt.nn.ReLU()
    self.relu3 = pt.nn.ReLU()

  def forward(self, x, hidden, cell_state, lengths):
    # Passing in the input(x) and hidden state into the model and obtaining the outputs
    # Use packed sequence to speed up the RNN/LSTM
    # pack_padded_sequence => RNN => pad_packed_sequence[0] to get the data in batch
    x_packed = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
    out_packed, (hidden, cell_state) = self.lstm1(x_packed, (hidden, cell_state))
    out_packed, (hidden, cell_state) = self.lstm2(out_packed, (hidden, cell_state))
    out_packed, (hidden, cell_state) = self.lstm3(out_packed, (hidden, cell_state))
    # Reshaping the output such that it can be fit into the fc layer
    out_unpacked = pad_packed_sequence(out_packed, batch_first=True, padding_value=-1)[0]
    out = self.fc1(out_unpacked)
    out = self.relu1(out)
    out = self.fc2(out)
    out = self.relu2(out)
    out = self.fc3(out)
    out = self.relu3(out)
    out = self.fc4(out)
    return out, (hidden, cell_state)

  def initHidden(self, batch_size):
    hidden = Variable(pt.randn(self.n_layers, batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return hidden

  def initCellState(self, batch_size):
    cell_state = Variable(pt.randn(self.n_layers, batch_size, self.hidden_dim, dtype=pt.float32)).cuda()
    return cell_state

