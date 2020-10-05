from __future__ import print_function
# Import libs
import numpy as np
np.set_printoptions(precision=4)
import torch as pt
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly
import plotly.graph_objects as go
import glob
import os
import argparse
import sys
import time
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
# Loss
import loss

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
  print('[%]GPU Enabled')
else:
  device = pt.device('cpu')
  print('[%]GPU Disabled, CPU Enabled')

# marker_dict for contain the marker properties
marker_dict_gt = dict(color='rgba(0, 0, 255, 0.7)', size=3)
marker_dict_pred = dict(color='rgba(255, 0, 0, 0.7)', size=3)
# Visualize the displacement
marker_dict_u = dict(color='rgba(255, 0, 0, 0.7)', size=4)
marker_dict_v = dict(color='rgba(0, 255, 0, 0.7)', size=4)
marker_dict_depth = dict(color='rgba(0, 0, 255, 0.5)', size=4)
marker_dict_eot = dict(color='rgba(0, 255, 0, 1)', size=5)

def make_visualize(input_test_dict, gt_test_dict, visualization_path, pred_test_dict, evaluation_results, animation_visualize_flag, args):
  # Visualize by make a subplots of trajectory
  n_vis = 5
  if n_vis > args.batch_size:
    n_vis = args.batch_size
  elif n_vis > input_test_dict['input'].shape[0]:
    n_vis = input_test_dict['input'].shape[0]
  # Random the index the be visualize
  vis_idx = np.random.choice(a=np.arange(input_test_dict['input'].shape[0]), size=(n_vis), replace=False)

  ####################################
  ############ Trajectory ############
  ####################################
  fig = make_subplots(rows=n_vis*2, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}], [{'colspan':2}, None]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  # Visualize a trajectory
  if args.env == 'unity':
    gt_eot = input_test_dict['input'][..., [2]]
  else:
    gt_eot = None

  fig = visualize_trajectory(uv=input_test_dict['input'], pred_xyz=pt.mul(pred_test_dict['xyz'], gt_test_dict['mask'][..., [0, 1, 2]]), gt_xyz=gt_test_dict['xyz'][..., [0, 1, 2]], startpos=gt_test_dict['startpos'], lengths=input_test_dict['lengths'], mask=gt_test_dict['mask'], fig=fig, flag='Test', n_vis=n_vis, evaluation_results=evaluation_results, vis_idx=vis_idx, pred_eot=pred_test_dict['flag'], gt_eot=gt_eot, args=args)
  # Adjust the layout/axis
  # AUTO SCALED/PITCH SCALED
  fig.update_layout(height=2048, width=1500, autosize=True, title="Testing on {} trajectory: Trajectory Visualization with EOT flag(Col1=PITCH SCALED, Col2=AUTO SCALED)".format(args.trajectory_type))
  fig = visualize_layout_update(fig=fig, n_vis=n_vis)
  plotly.offline.plot(fig, filename='./{}/trajectory_visualization_depth.html'.format(args.visualization_path), auto_open=True)
  if animation_visualize_flag:
    trajectory_animation(output_xyz=pt.mul(pred_test_dict['xyz'], gt_test_dict['mask'][..., [0, 1, 2]]), gt_xyz=gt_test_dict['xyz'][..., [0, 1, 2]], input_uv=input_test_dict['input'], lengths=input_test_dict['lengths'], mask=gt_test_dict['mask'][..., [0, 1, 2]], n_vis=n_vis, html_savepath=visualization_path, vis_idx=vis_idx)
  input("Continue plotting...")

'''
For perspective projection
# def visualize_layout_update(fig=None, n_vis=7):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  # fig.update_layout(height=1920, width=1080, margin=dict(l=0, r=0, b=5,t=5,pad=1), autosize=False)
  # for i in range(n_vis*2):
    # if i%2==0:
      # Set the figure in column 1 (fig0, 2, 4, ...) into a pitch scaled
      # fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-5, 5],), yaxis = dict(nticks=5, range=[-2, 4],), zaxis = dict(nticks=10, range=[-5, 5],),)
      # fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-27, 33],), yaxis = dict(nticks=5, range=[-2, 12],), zaxis = dict(nticks=10, range=[-31, 19],), aspectmode='manual', aspectratio=dict(x=4, y=2, z=3))
    # fig['layout']['scene{}'.format(i+1)]['camera'].update(projection=dict(type="perspective"))
  # return fig
'''

def visualize_layout_update(fig=None, n_vis=3):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  for i in range(n_vis*2):
    fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-27, 33],), yaxis = dict(nticks=5, range=[-50, 12],), zaxis = dict(nticks=10, range=[-31, 19],), aspectmode='manual', aspectratio=dict(x=4, y=2, z=3))
  return fig

def visualize_trajectory(uv, pred_xyz, gt_xyz, startpos, lengths, mask, evaluation_results, vis_idx, gt_eot, pred_eot, args, fig=None, flag='test', n_vis=5):
  # detach() for visualization
  uv = uv.cpu().detach().numpy()
  pred_eot = pred_eot.cpu().detach().numpy()
  pred_xyz = pred_xyz.cpu().detach().numpy()
  gt_xyz = gt_xyz.cpu().detach().numpy()
  if args.env == 'unity':
    gt_eot = gt_eot.cpu().detach().numpy()
  # Iterate to plot each trajectory
  count = 1
  for idx, i in enumerate(vis_idx):
    for col_idx in range(1, 3):
      fig.add_trace(go.Scatter3d(x=pred_xyz[i][:lengths[i]+1, 0], y=pred_xyz[i][:lengths[i]+1, 1], z=pred_xyz[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}, MAE_trajectory = {}, MaxDist = {}".format(flag, i, loss.TrajectoryLoss(pt.tensor(pred_xyz[i]).to(device), pt.tensor(gt_xyz[i]).to(device), mask=mask[i]), evaluation_results['MAE']['loss_3axis'][i], evaluation_results['MAE']['maxdist_3axis'][i, :])), row=idx+count, col=col_idx)
      fig.add_trace(go.Scatter3d(x=gt_xyz[i][:lengths[i]+1, 0], y=gt_xyz[i][:lengths[i]+1, 1], z=gt_xyz[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+count, col=col_idx)
    count +=1
  # Iterate to plot each displacement of (u, v, depth)
  for idx, i in enumerate(vis_idx):
    col_idx = 1
    row_idx = (idx*2) + 2
    fig.add_trace(go.Scatter(x=np.arange(pred_eot[i][:lengths[i]].shape[0]), y=pred_eot[i][:lengths[i]].reshape(-1), marker=marker_dict_eot, mode='markers+lines', name='{}-Trajectory [{}], EOT PRED'.format(flag, i)), row=row_idx, col=col_idx)
    fig.add_trace(go.Scatter(x=np.arange(uv[i][:lengths[i], 0].shape[0]), y=uv[i][:lengths[i]+1, 0], marker=marker_dict_gt, mode='lines', name='{}-Trajectory [{}], U'.format(flag, i)), row=row_idx, col=col_idx)
    fig.add_trace(go.Scatter(x=np.arange(uv[i][:lengths[i], 1].shape[0]), y=uv[i][:lengths[i]+1, 1], marker=marker_dict_gt, mode='lines', name='{}-Trajectory [{}], V'.format(flag, i)), row=row_idx, col=col_idx)

    if args.env == 'unity':
      fig.add_trace(go.Scatter(x=np.arange(gt_eot[i][:lengths[i]].shape[0]), y=gt_eot[i][:lengths[i]].reshape(-1), marker=marker_dict_gt, mode='markers+lines', name='{}-Trajectory [{}], EOT GT'.format(flag, i)), row=row_idx, col=col_idx)
  return fig
