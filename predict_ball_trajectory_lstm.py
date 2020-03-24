from __future__ import print_function
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import glob
import os
import argparse
from tqdm import tqdm
from rnn_model import RNN
from lstm_model import LSTM
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from dataloader import TrajectoryDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import PIL.Image
import io
import wandb
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from mpl_toolkits import mplot3d

def predict(trajectory_gt, initial_condition_gt, model, visualize_trajectory_flag=True, writer=None):
  trajectory_gt_unpadded, initial_condition_gt_unpadded = np.copy(unpadded_tensor(np.copy(trajectory_gt.cpu().detach().clone().numpy()), np.copy(initial_condition_gt.cpu().detach().clone().numpy())))
  # Trajectory size = (#n trajectory, #seq_length, #n_output_coordinates)
  output_pred = np.copy(initial_condition_gt_unpadded)
  # output_pred = np.insert(output_pred, output_pred.shape[1], values=[-10, -10], axis=1)
  # Initial condition size = (#n trajectory, #seq_length, #n_input_coordinates)delta in vector space
  initial_condition_pred = np.copy(initial_condition_gt_unpadded)
  # Loop over every trajectory
  loss_fn = pt.nn.MSELoss()
  n_prior_point = 15
  model.eval()
  with pt.no_grad():
    for i in tqdm(range(trajectory_gt_unpadded.shape[0]), desc='Prediction Trajectory'):
      # Loop over length of the trajectory
      batch_size=1
      hidden = model.initHidden(batch_size=batch_size)
      cell_state = model.initCellState(batch_size=batch_size)
      # print(trajectory_gt_unpadded[i].shape)
      # print(initial_condition_gt_unpadded[i].shape)
      # print(output_pred[i].shape)
      for j in range(n_prior_point, trajectory_gt_unpadded[i].shape[0]):
        # print('All points {} : From {} to {}'.format(trajectory_gt_unpadded[i].shape[0], j-n_prior_point, j))
        # Init the initial_condition_pred from initial_condition_gt for the beginning of the trajectory
        # Make a prediction
        output, (hidden, cell_state) = model(pt.from_numpy(output_pred[i][j-n_prior_point:j].reshape(1, n_prior_point, 2)).cuda().float(), hidden, cell_state)
        try:
          output_pred[i][j] = np.copy(output[-1][:].cpu().detach().clone().numpy())
        except IndexError:
          output_pred[i] = np.vstack((output_pred[i], output[-1][:].cpu().detach().clone().numpy()))

        # print("LAST : ", model(initial_condition_pred[i][j-n_prior_point:j].reshape(1, n_prior_point, 2).float())[0][-1][:])
        # print("FIRST : ", model(initial_condition_pred[i][j-n_prior_point:j].reshape(1, n_prior_point, 2).float())[0][:][:])
        # Change the current point from predicted point not the ground truth point
        # initial_condition_pred[i][j][0] = np.copy(output_pred[i][j][0])
        # initial_condition_pred[i][j][1] = np.copy(output_pred[i][j][1])
        # print('=============={}=============='.format(j))
        # print('Input : ', output_pred[i][j-n_prior_point:j])
        # print('Prediction : ', output_pred[i][j])
        # print('Ground Truth : ', trajectory_gt[i][j-1])
      # print('Loss : ', loss_fn(pt.from_numpy(output_pred[i][:]).cuda().float(), pt.from_numpy(trajectory_gt_unpadded[i][:]).cuda().float()))
    if visualize_trajectory_flag == True:
      output_pred = np.array([np.cumsum(output_pred[i], axis=0)  for i in range(len(output_pred))])
      trajectory_gt_unpadded = np.array([np.cumsum(trajectory_gt_unpadded[i], axis=0)  for i in range(len(trajectory_gt_unpadded))])
      traj_pred_img = visualize_trajectory(output_pred, trajectory_gt_unpadded, writer=writer, n_vis=32)
      writer.add_image('Testing set : Trajectory Estimation', traj_pred_img, dataformats='NCHW')

