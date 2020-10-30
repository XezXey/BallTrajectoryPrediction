from __future__ import print_function
# Import libs
import numpy as np
np.set_printoptions(precision=4)
import torch as pt
import glob
import os
import argparse
import sys
import time
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly
import plotly.graph_objects as go
sys.path.append(os.path.realpath('../..'))
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import PIL.Image
import io
from sklearn.metrics import confusion_matrix
import wandb
import json
# Utils
from utils.dataloader import TrajectoryDataset
import utils.utils_func as utils_func
import utils.cummulative_depth as utils_cummulative
import utils.transformation as utils_transform
import utils.utils_inference_func as utils_inference_func
# Loss
import loss

# marker_dict for contain the marker properties
marker_dict_gt = dict(color='rgba(0, 0, 255, 0.7)', size=3)
marker_dict_pred = dict(color='rgba(255, 0, 0, 0.7)', size=3)
# Visualize the displacement
marker_dict_u = dict(color='rgba(255, 0, 0, 0.7)', size=4)
marker_dict_v = dict(color='rgba(0, 255, 0, 0.7)', size=4)
marker_dict_depth = dict(color='rgba(0, 0, 255, 0.5)', size=4)
marker_dict_eot = dict(color='rgba(0, 255, 0, 1)', size=5)
# Argumentparser for input
parser = argparse.ArgumentParser(description='Visualize and ranking the 3D projectile')
parser.add_argument('--loadfromfile', dest='loadfromfile', type=str, help='Path to load prediction', default=None)
parser.add_argument('--rank', dest='rank', type=str, help='Ranking by', default='MSE')
parser.add_argument('--order', dest='order', type=str, help='Ranking order (a=ascending, d=descending)', default='a')
args = parser.parse_args()
# Prediction folder prefix
prediction_file = args.loadfromfile.split('/')[-2]


def ranking(eval_metrices, rank_by, lengths):
  # Rank by
  if rank_by == 'MSE' or rank_by == 'MAE':
    mean = np.mean(eval_metrices[rank_by]['loss_3axis'], axis=1)
    ranking_index = np.argsort(a=mean)
  elif rank_by == 'seq_len':
    ranking_index = np.argsort(a=lengths)
  # Sort by
  if args.order == 'd': # Descending order
    ranking_index = np.flip(ranking_index)

  return ranking_index

def visualize(trajectory, eval_metrices, lengths, n_plot, plot_idx):
  fig = make_subplots(rows=n_plot*2, cols=1, specs=[[{'type':'scatter3d'}], [{'type':'scatter'}]]*n_plot, horizontal_spacing=0.01, vertical_spacing=0.01)
  fig.update_layout(height=4096, width=1800, autosize=False, title="Trajectory Visualization")
  for i in range(n_plot):
    row_idx = (i*2)
    ####################################
    ############ Trajectory ############
    ####################################
    gt_xyz = trajectory[i][:, [0, 1, 2]]
    pred_xyz = trajectory[i][:, [3, 4, 5]]
    uvd = trajectory[i][:, [6, 7, 8]]
    seq_len = lengths[i]
    fig.add_trace(go.Scatter3d(x=gt_xyz[:, 0], y=gt_xyz[:, 1], z=gt_xyz[:, 2], mode='markers', marker=marker_dict_gt, name="Ground Truth Trajectory".format()), row=row_idx+1, col=1)
    fig.add_trace(go.Scatter3d(x=pred_xyz[:, 0], y=pred_xyz[:, 1], z=pred_xyz[:, 2], mode='markers', marker=marker_dict_pred, name="Prediction Trajectory {}={:3f} (Each axis={})".format(args.rank, np.mean(eval_metrices[i]), eval_metrices[i])), row=row_idx+1, col=1)
    # fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-3, 4]), yaxis = dict(nticks=10, range=[-2, 2]), zaxis = dict(nticks=10, range=[-2, 2]), aspectmode='cube',)# aspectratio=dict(x=4, y=4, z=4))
    # fig['layout']['scene{}'.format(i+1)].update(aspectmode='cube', camera=dict(eye=dict(x=-0.2, y=-3.8, z=3.8)))# aspectratio=dict(x=4, y=4, z=4))
    ####################################
    ########### Displacement ###########
    ####################################
    fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=np.diff(uvd[:, 0]), mode='markers+lines', marker=marker_dict_u, name="U"), row=row_idx+2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=np.diff(uvd[:, 1]), mode='markers+lines', marker=marker_dict_v, name="V"), row=row_idx+2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=np.diff(uvd[:, 2]), mode='markers+lines', marker=marker_dict_depth, name="Depth"), row=row_idx+2, col=1)

  plotly.offline.plot(fig, filename='./{}/reconstructed_trajectory_{}.html'.format(args.loadfromfile, plot_idx), auto_open=False)

if __name__ == '__main__':
  trajectory = np.load('{}/{}_trajectory.npy'.format(args.loadfromfile, prediction_file), allow_pickle=True)
  eval_metrices = np.load('{}/{}_metrices.npy'.format(args.loadfromfile, prediction_file), allow_pickle=True)

  # Try catch
  if args.rank not in ['MAE', 'MSE', 'seq_len']:
    print("[#] Please rank by MAE, MSE or seq_len")
    exit()

  # Lengths
  lengths = np.array([each_traj.shape[0] for each_traj in trajectory])
  # Ranking
  ranking_index = ranking(eval_metrices=eval_metrices[()], rank_by=args.rank, lengths=lengths)
  # Plotting
  n_plot = 5    # Plotting 7 trajectory per page
  n_trajectory = trajectory.shape[0]
  for i in range((n_trajectory//n_plot) + 1):
    # Start-End index
    start = i * n_plot
    end = start + n_plot
    if end > n_trajectory:
      end = n_trajectory
      n_plot = end - start
    if end == start:
      break
    print("Start : ", start, ", End : ", end)
    if args.rank == 'seq_len':
      args.rank = 'MSE'
      metrices = eval_metrices[()]['MSE']['loss_3axis'][start:end]
    elif args.rank == 'MSE' or args.rank =='MAE':
      metrices = eval_metrices[()][args.rank]['loss_3axis'][start:end]

    visualize(trajectory=trajectory[ranking_index[start:end]], eval_metrices=metrices, lengths=lengths[ranking_index[start:end]], n_plot=n_plot, plot_idx=i)

