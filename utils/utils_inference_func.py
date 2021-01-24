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
import math
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
import utils.loss as utils_loss

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
  print('[%]GPU Enabled')
else:
  device = pt.device('cpu')
  print('[%]GPU Disabled, CPU Enabled')

args = None
def share_args(a):
  global args
  args = a

# marker_dict for contain the marker properties
marker_dict_gt = dict(color='rgba(0, 0, 255, 0.7)', size=3)
marker_dict_pred = dict(color='rgba(255, 0, 0, 0.7)', size=3)
# Visualize the displacement
marker_dict_u = dict(color='rgba(255, 0, 0, 0.7)', size=4)
marker_dict_v = dict(color='rgba(0, 255, 0, 0.7)', size=4)
marker_dict_depth = dict(color='rgba(0, 0, 255, 0.5)', size=4)
marker_dict_latent = dict(color='rgba(11, 102, 35, 0.7)', size=7)
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
  if args.env == 'unity' and 'eot' in args.pipeline:
    gt_eot = input_test_dict['input'][..., [2]]
  else:
    gt_eot = None

  print("VIS : ", pred_test_dict['xyz'])
  fig = visualize_trajectory(uv=input_test_dict['input'], pred_xyz=pt.mul(pred_test_dict['xyz'][..., [0, 1, 2]], gt_test_dict['mask'][..., [0, 1, 2]]), gt_xyz=gt_test_dict['xyz'][..., [0, 1, 2]], startpos=gt_test_dict['startpos'], lengths=input_test_dict['lengths'], mask=gt_test_dict['mask'], fig=fig, flag='Test', n_vis=n_vis, evaluation_results=evaluation_results, vis_idx=vis_idx, pred_eot=pred_test_dict['flag'], gt_eot=gt_eot, args=args, latent=pred_test_dict['xyz'][..., 3:])
  # Adjust the layout/axis
  # AUTO SCALED/PITCH SCALED
  fig.update_layout(height=2048, width=2048, autosize=False, title="Testing on {} trajectory: Trajectory Visualization with EOT flag(Col1=PITCH SCALED, Col2=AUTO SCALED)".format(args.trajectory_type))
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
    if i%2==0:
      # fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-27, 33],), yaxis = dict(nticks=5, range=[-3, 12],), zaxis = dict(nticks=10, range=[-31, 19],), aspectmode='manual', aspectratio=dict(x=4, y=2, z=3))
      # fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-4, 4],), yaxis = dict(nticks=5, range=[-3, 8],), zaxis = dict(nticks=10, range=[-4, 4],), aspectmode='manual', aspectratio=dict(x=4, y=2, z=3))
      fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(dtick=1, range=[-4, 4],), yaxis = dict(dtick=1, range=[-4, 4],), zaxis = dict(dtick=1, range=[-4, 4]), aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
                                                  camera=dict(eye=dict(x=0.2, y=3.9, z=3.9),
                                                              up=dict(x=0, y=1, z=0)))
      # fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(dtick=1, range=[-50, 50],), yaxis = dict(dtick=1, range=[-6, 6],), zaxis = dict(dtick=1, range=[-40, 40]), aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
                                                  # camera=dict(eye=dict(x=0.2, y=3.9, z=3.9),
                                                              # up=dict(x=0, y=1, z=0)))
  return fig

def visualize_trajectory(uv, pred_xyz, gt_xyz, startpos, lengths, mask, evaluation_results, vis_idx, gt_eot, pred_eot, args, latent, fig=None, flag='test', n_vis=5):
  # detach() for visualization
  uv = uv.cpu().detach().numpy()
  pred_xyz = pred_xyz.cpu().detach().numpy()
  gt_xyz = gt_xyz.cpu().detach().numpy()
  # latent = latent / pt.sqrt(pt.sum(latent**2, dim=1, keepdims=True) + 1e-16)
  latent = latent.cpu().detach().numpy()
  if pred_eot is not None:
    pred_eot = pred_eot.cpu().detach().numpy()
    eot = np.concatenate((np.zeros((pred_eot.shape[0], 1, 1)), pred_eot), axis=1)
    close = np.isclose(eot, np.array([1.]), atol=5e-1)
  if gt_eot is not None:
    gt_eot = gt_eot.cpu().detach().numpy()
  # Iterate to plot each trajectory
  count = 1
  for idx, i in enumerate(vis_idx):
    for col_idx in range(1, 3):
      fig.add_trace(go.Scatter3d(x=pred_xyz[i][:lengths[i]+1, 0], y=pred_xyz[i][:lengths[i]+1, 1], z=-pred_xyz[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}, MAE_trajectory = {}, MaxDist = {}".format(flag, i, utils_loss.TrajectoryLoss(pt.tensor(pred_xyz[i]).to(device), pt.tensor(gt_xyz[i]).to(device), mask=mask[i]), evaluation_results['MAE']['loss_3axis'][i], evaluation_results['MAE']['maxdist_3axis'][i, :])), row=idx+count, col=col_idx)
      fig.add_trace(go.Scatter3d(x=gt_xyz[i][:lengths[i]+1, 0], y=gt_xyz[i][:lengths[i]+1, 1], z=-gt_xyz[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+count, col=col_idx)
      if args.optimize:
        where = np.where(close[i] == True)[0]
        where = where[where < lengths[i].cpu().detach().numpy()]
        if len(where) == 0:
          where = [0]
        else:
          where = [0] + list(where)
        print(latent[i])
        for latent_pos in where:
          if 'angle' in args.latent_code:
            # Latent size = 1 (Optimize angle)
            latent_arrow_x = np.array([pred_xyz[i][latent_pos, 0], pred_xyz[i][latent_pos, 0] + np.cos(np.abs(latent[i][latent_pos, 0]) * math.pi/180.0) * 10])
            latent_arrow_y = np.array([pred_xyz[i][latent_pos, 1], pred_xyz[i][latent_pos, 1]])
            latent_arrow_z = np.array([pred_xyz[i][latent_pos, 2], pred_xyz[i][latent_pos, 2] + np.sin(np.abs(latent[i][latent_pos, 0]) * math.pi/180.0) * 10])
          if 'sin_cos' in args.latent_code:
            # Latent size = 2 (Optimize sin_cos directly)
            latent[i] = latent[i] / (np.sqrt(np.sum(latent[i]**2, axis=1, keepdims=True)) + 1e-16)
            latent_arrow_x = np.array([pred_xyz[i][latent_pos, 0], pred_xyz[i][latent_pos, 0] + latent[i][latent_pos, 0] * 10])
            latent_arrow_y = np.array([pred_xyz[i][latent_pos, 1], pred_xyz[i][latent_pos, 1]])
            latent_arrow_z = np.array([pred_xyz[i][latent_pos, 2], pred_xyz[i][latent_pos, 2] + latent[i][latent_pos, 1] * 10])
          else:
            # Latent size = 3 (Optimize Force direction)
            latent_arrow_x = [pred_xyz[i][latent_pos, 0], pred_xyz[i][latent_pos, 0]+latent[i][latent_pos, 0]*10]
            latent_arrow_y = [pred_xyz[i][latent_pos, 1], pred_xyz[i][latent_pos, 1]+latent[i][latent_pos, 1]*10]
            latent_arrow_z = [pred_xyz[i][latent_pos, 2], pred_xyz[i][latent_pos, 2]+latent[i][latent_pos, 2]*10]
          fig.add_trace(go.Scatter3d(x=latent_arrow_x, y=latent_arrow_y, z=-latent_arrow_z, mode='lines', line=dict(width=10), marker=marker_dict_latent, name="{}-Optimized Latent [{}]".format(flag, i)), row=idx+count, col=col_idx)
        # fig.add_trace(go.Cone(x=pred_xyz[i][:lengths[i]+1, 0], y=pred_xyz[i][:lengths[i]+1, 1], z=pred_xyz[i][:lengths[i]+1, 2], u=latent[i][:lengths[i]+1, 0]*1000, v=latent[i][:lengths[i]+1, 1]*1000, w=latent[i][:lengths[i]+1, 2]*1000, showscale=False, sizeref=20, opacity=0.5, name="{}-Optimized Latent [{}]".format(flag, i), colorscale='Greens'), row=idx+count, col=col_idx)
    count +=1
  # Iterate to plot each displacement of (u, v, depth)
  for idx, i in enumerate(vis_idx):
    col_idx = 1
    row_idx = (idx*2) + 2
    fig.add_trace(go.Scatter(x=np.arange(uv[i][:lengths[i], 0].shape[0]), y=uv[i][:lengths[i]+1, 0], marker=marker_dict_gt, mode='lines', name='{}-Trajectory [{}], U'.format(flag, i)), row=row_idx, col=col_idx)
    fig.add_trace(go.Scatter(x=np.arange(uv[i][:lengths[i], 1].shape[0]), y=uv[i][:lengths[i]+1, 1], marker=marker_dict_gt, mode='lines', name='{}-Trajectory [{}], V'.format(flag, i)), row=row_idx, col=col_idx)

    if pred_eot is not None:
      fig.add_trace(go.Scatter(x=np.arange(pred_eot[i][:lengths[i]].shape[0]), y=pred_eot[i][:lengths[i]].reshape(-1), marker=marker_dict_eot, mode='markers+lines', name='{}-Trajectory [{}], EOT PRED'.format(flag, i)), row=row_idx, col=col_idx)
    if gt_eot is not None:
      fig.add_trace(go.Scatter(x=np.arange(gt_eot[i][:lengths[i]].shape[0]), y=gt_eot[i][:lengths[i]].reshape(-1), marker=marker_dict_gt, mode='markers+lines', name='{}-Trajectory [{}], EOT GT'.format(flag, i)), row=row_idx, col=col_idx)
  return fig