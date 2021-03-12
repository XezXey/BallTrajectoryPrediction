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
marker_dict_input = dict(color='rgba(0, 128, 0, 0.7)', size=3)
marker_dict_gt = dict(color='rgba(0, 0, 255, 0.7)', size=3)
marker_dict_pred = dict(color='rgba(255, 0, 0, 0.7)', size=3)
# Visualize the displacement
marker_dict_u = dict(color='rgba(255, 0, 0, 0.7)', size=4)
marker_dict_v = dict(color='rgba(0, 255, 0, 0.7)', size=4)
marker_dict_depth = dict(color='rgba(0, 0, 255, 0.5)', size=4)
marker_dict_latent = dict(color='rgba(11, 102, 35, 0.7)', size=7)
marker_dict_eot = dict(color='rgba(0, 255, 0, 1)', size=5)

def autoregressive_plot(pred_uv, gt_dict, ):
  # Here we passed input_test_dict as a gt_dict.
  gt_uv = pt.cumsum(pt.cat((gt_dict['startpos'][..., [0, 1]], gt_dict['input'][..., [0, 1]]), dim=1), dim=1).detach().cpu().numpy()
  pred_uv = pt.cumsum(pt.cat((gt_dict['startpos'][..., [0, 1]], pred_uv), dim=1), dim=1).detach().cpu().numpy()
  # missing = pt.cat((pt.zeros(size=(pred_uv.shape[0], 1, 1)).long().cuda(), gt_dict['in_f_missing_duv'].cuda(), pt.zeros(size=(pred_uv.shape[0], 1, 1)).long().cuda()), dim=1).detach().cpu().numpy()
  # missing = missing[:, :-1, :] & missing[:, 1:, :]
  missing = gt_dict['in_f_missing_uv'].detach().cpu().numpy()
  missing = np.where(missing == 1, np.nan, missing)
  lengths = gt_dict['lengths']+1
  # Visualize by make a subplots of trajectory
  n_vis = 5
  if n_vis > args.batch_size:
    n_vis = args.batch_size
  elif n_vis > gt_dict['input'].shape[0]:
    n_vis = gt_dict['input'].shape[0]

  # Random the index the be visualize
  vis_idx = np.random.choice(a=np.arange(gt_dict['input'].shape[0]), size=(n_vis), replace=False)
  fig = make_subplots(rows=n_vis, cols=1, specs=[[{'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)

  for idx, i in enumerate(vis_idx):
    missing_x = (missing[i][:lengths[i], [0]] * gt_uv[i][:lengths[i], [0]] + gt_uv[i][:lengths[i], [0]]).reshape(-1)
    missing_y = (missing[i][:lengths[i], [0]] * gt_uv[i][:lengths[i], [1]] + gt_uv[i][:lengths[i], [1]]).reshape(-1)
    fig.add_trace(go.Scatter(x=gt_uv[i][:lengths[i], 0], y=gt_uv[i][:lengths[i], 1], mode='markers+lines', marker=marker_dict_gt, name='Traj#{}-UV-GroundTruth'.format(i)), row=idx+1, col=1)
    fig.add_trace(go.Scatter(x=missing_x, y=missing_y, mode='markers+lines', marker=marker_dict_input, name='Traj#{}-UV-Input'.format(i)), row=idx+1, col=1)
    fig.add_trace(go.Scatter(x=pred_uv[i][:lengths[i], 0], y=pred_uv[i][:lengths[i], 1], mode='markers+lines', marker=marker_dict_pred, name='Traj#{}-UV-Interpolated'.format(i)), row=idx+1, col=1)

  plotly.offline.plot(fig, filename='./{}/trajectory_visualization_ar.html'.format(args.visualization_path), auto_open=True)

def make_visualize(input_test_dict, gt_test_dict, visualization_path, pred_test_dict, evaluation_results, animation_visualize_flag, args, cam_params_dict):
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

  fig = visualize_trajectory(uv=input_test_dict['input'], pred_xyz=pt.mul(pred_test_dict['xyz'][..., [0, 1, 2]], gt_test_dict['mask'][..., [0, 1, 2]]), gt_xyz=gt_test_dict['xyz'][..., [0, 1, 2]], startpos=gt_test_dict['startpos'], lengths=input_test_dict['lengths'], mask=gt_test_dict['mask'], fig=fig, flag='Test', n_vis=n_vis, evaluation_results=evaluation_results, vis_idx=vis_idx, pred_eot=pred_test_dict['flag'], gt_eot=gt_eot, args=args, latent_optimized=pred_test_dict['latent_optimized'], cam_params_dict=cam_params_dict['main'])
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
      fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(dtick=1, range=[-4, 4],), yaxis = dict(dtick=1, range=[-4, 4],), zaxis = dict(dtick=1, range=[-4, 4]), aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
                                                  camera=dict(eye=dict(x=0.2, y=3.9, z=3.9),
                                                              up=dict(x=0, y=1, z=0)))
      # fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(dtick=1, range=[-50, 50],), yaxis = dict(dtick=1, range=[-6, 6],), zaxis = dict(dtick=1, range=[-40, 40]), aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
                                                  # camera=dict(eye=dict(x=0.2, y=3.9, z=3.9),
                                                              # up=dict(x=0, y=1, z=0)))
  return fig

def visualize_trajectory(uv, pred_xyz, gt_xyz, startpos, lengths, mask, evaluation_results, vis_idx, gt_eot, pred_eot, args, latent_optimized, cam_params_dict, fig=None, flag='test', n_vis=5):
  # detach() for visualization
  uv = uv.cpu().detach().numpy()
  gt_xyz = gt_xyz.clone().cpu().detach().numpy()
  # gt_xyz = np.where(np.isclose(0.0, gt_xyz, atol=1e-6), np.nan, gt_xyz)

  pred_xyz = pred_xyz.cpu().detach().numpy()

  if args.optimize is not None:
    latent_optimized = latent_optimized.cpu().detach().numpy()
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
      fig.add_trace(go.Scatter3d(x=pred_xyz[i][:lengths[i]+1, 0], y=pred_xyz[i][:lengths[i]+1, 1], z=pred_xyz[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}, MAE_trajectory = {}, MaxDist = {}".format(flag, i, utils_loss.TrajectoryLoss(pt.tensor(pred_xyz[i]).to(device), pt.tensor(gt_xyz[i]).to(device), mask=mask[i]), evaluation_results['MAE']['loss_3axis'][i], evaluation_results['MAE']['maxdist_3axis'][i, :])), row=idx+count, col=col_idx)
      fig.add_trace(go.Scatter3d(x=gt_xyz[i][:lengths[i]+1, 0], y=gt_xyz[i][:lengths[i]+1, 1], z=gt_xyz[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+count, col=col_idx)
      if args.optimize is not None:
        # We did an optimization on Depth or Refinement network
        where = np.where(close[i] == True)[0]
        where = where[where < lengths[i].cpu().detach().numpy()]
        if len(where) == 0:
          where = [0]
        else:
          where = [0] + list(where)
        for latent_pos in where:
          if 'angle' in args.latent_code:
            # Latent size = 1 (Optimize angle)
            latent_arrow_x = np.array([pred_xyz[i][latent_pos, 0], pred_xyz[i][latent_pos, 0] + np.cos(np.abs(latent_optimized[i][latent_pos, 0]) * math.pi/180.0)])
            latent_arrow_y = np.array([pred_xyz[i][latent_pos, 1], pred_xyz[i][latent_pos, 1]])
            latent_arrow_z = np.array([pred_xyz[i][latent_pos, 2], pred_xyz[i][latent_pos, 2] + np.sin(np.abs(latent_optimized[i][latent_pos, 0]) * math.pi/180.0)])
          elif 'sin_cos' in args.latent_code:
            # Latent size = 2 (Optimize sin_cos directly)
            latent_optimized[i] = latent_optimized[i] / (np.sqrt(np.sum(latent_optimized[i]**2, axis=-1, keepdims=True)) + 1e-16)
            latent_arrow_x = np.array([pred_xyz[i][latent_pos, 0], pred_xyz[i][latent_pos, 0] + latent_optimized[i][latent_pos, 1]])
            latent_arrow_y = np.array([pred_xyz[i][latent_pos, 1], pred_xyz[i][latent_pos, 1]])
            latent_arrow_z = np.array([pred_xyz[i][latent_pos, 2], pred_xyz[i][latent_pos, 2] + latent_optimized[i][latent_pos, 0]])
          elif 'f' in args.latent_code:
            latent_optimized[i] = latent_optimized[i] / (np.sqrt(np.sum(latent_optimized[i]**2, axis=-1, keepdims=True)) + 1e-16)
            # Latent size = 3 (Optimize Force direction)
            latent_arrow_x = [pred_xyz[i][latent_pos, 0], pred_xyz[i][latent_pos, 0] + latent_optimized[i][latent_pos, 0]]
            latent_arrow_y = [pred_xyz[i][latent_pos, 1], pred_xyz[i][latent_pos, 1] + latent_optimized[i][latent_pos, 1]]
            latent_arrow_z = [pred_xyz[i][latent_pos, 2], pred_xyz[i][latent_pos, 2] + latent_optimized[i][latent_pos, 2]]
          elif 'f_norm' in args.latent_code:
            # Latent size = 3 (Optimize Force direction)
            latent_arrow_x = [pred_xyz[i][latent_pos, 0], pred_xyz[i][latent_pos, 0] + latent_optimized[i][latent_pos, 0]]
            latent_arrow_y = [pred_xyz[i][latent_pos, 1], pred_xyz[i][latent_pos, 1] + latent_optimized[i][latent_pos, 1]]
            latent_arrow_z = [pred_xyz[i][latent_pos, 2], pred_xyz[i][latent_pos, 2] + latent_optimized[i][latent_pos, 2]]
          fig.add_trace(go.Scatter3d(x=latent_arrow_x, y=latent_arrow_y, z=latent_arrow_z, mode='lines', line=dict(width=10), marker=marker_dict_latent, name="{}-Optimized Latent [{}]".format(flag, i)), row=idx+count, col=col_idx)
    count +=1
  # Iterate to plot each displacement of (u, v, depth)
  for idx, i in enumerate(vis_idx):
    col_idx = 1
    row_idx = (idx*2) + 2
    # Projection
    pred_xyz_proj = utils_transform.projectToScreenSpace(world=pt.tensor(pred_xyz[[i], ...]).to(device), cam_params_dict=cam_params_dict, normalize=False)
    gt_xyz_proj = utils_transform.projectToScreenSpace(world=pt.tensor(gt_xyz[[i], ...]).to(device), cam_params_dict=cam_params_dict, normalize=False)
    uv_pred_proj = pt.cat((pred_xyz_proj[0], pred_xyz_proj[1]), dim=2).cpu().detach().numpy()
    uv_gt_proj = pt.cat((gt_xyz_proj[0], gt_xyz_proj[1]), dim=2).cpu().detach().numpy()

    duv_pred_proj = uv_pred_proj[:, 1:, :] - uv_pred_proj[:, :-1, :]
    duv_gt_proj = uv_gt_proj[:, 1:, :] - uv_gt_proj[:, :-1, :]

    fig.add_trace(go.Scatter(x=np.arange(duv_gt_proj[0][:lengths[i], 0].shape[0]-1), y=duv_gt_proj[0][:lengths[i]-1, 0], marker=marker_dict_gt, mode='lines', name='{}-Trajectory [{}], dU-GT'.format(flag, i)), row=row_idx, col=col_idx)
    fig.add_trace(go.Scatter(x=np.arange(duv_gt_proj[0][:lengths[i], 1].shape[0]-1), y=duv_gt_proj[0][:lengths[i]-1, 1], marker=marker_dict_gt, mode='lines', name='{}-Trajectory [{}], dV-GT'.format(flag, i)), row=row_idx, col=col_idx)
    fig.add_trace(go.Scatter(x=np.arange(duv_pred_proj[0][:lengths[i], 0].shape[0]-1), y=duv_pred_proj[0][:lengths[i]-1, 0], marker=marker_dict_pred, mode='lines', name='{}-Trajectory [{}], dU-PRED'.format(flag, i)), row=row_idx, col=col_idx)
    fig.add_trace(go.Scatter(x=np.arange(duv_pred_proj[0][:lengths[i], 1].shape[0]-1), y=duv_pred_proj[0][:lengths[i]-1, 1], marker=marker_dict_pred, mode='lines', name='{}-Trajectory [{}], dV-PRED'.format(flag, i)), row=row_idx, col=col_idx)

    fig.add_trace(go.Scatter(x=uv_gt_proj[0][:lengths[i], 0], y=uv_gt_proj[0][:lengths[i], 1], marker=marker_dict_gt, mode='lines', name='{}-Trajectory [{}], UV-Gt'.format(flag, i)), row=row_idx, col=col_idx)
    fig.add_trace(go.Scatter(x=uv_pred_proj[0][:lengths[i], 0], y=uv_pred_proj[0][:lengths[i], 1], marker=marker_dict_pred, mode='lines', name='{}-Trajectory [{}], UV-Pred'.format(flag, i)), row=row_idx, col=col_idx)

    if pred_eot is not None:
      fig.add_trace(go.Scatter(x=np.arange(pred_eot[i][:lengths[i]].shape[0]), y=pred_eot[i][:lengths[i]].reshape(-1), marker=marker_dict_eot, mode='markers+lines', name='{}-Trajectory [{}], EOT PRED'.format(flag, i)), row=row_idx, col=col_idx)
    if gt_eot is not None:
      fig.add_trace(go.Scatter(x=np.arange(gt_eot[i][:lengths[i]].shape[0]), y=gt_eot[i][:lengths[i]].reshape(-1), marker=marker_dict_gt, mode='markers+lines', name='{}-Trajectory [{}], EOT GT'.format(flag, i)), row=row_idx, col=col_idx)
  return fig
