from __future__ import print_function
# Import libs
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import glob
import os
import argparse
import sys
sys.path.append(os.path.realpath('../..'))
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import confusion_matrix
import PIL.Image
import io
# from wandb import magic
import wandb
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from mpl_toolkits import mplot3d
import json
# Dataloader
from utils.dataloader import TrajectoryDataset
# Models
from models.Simple.rnn_model import RNN
from models.Simple.lstm_model import LSTM
from models.Simple.bilstm_model import BiLSTM
from models.Simple.gru_model import GRU
from models.Simple.bigru_model import BiGRU
from models.Simple.bigru_model_residual_list import BiGRUResidualList
from models.Simple.bigru_model_residual_add import BiGRUResidualAdd
from torch.utils.tensorboard import SummaryWriter

def visualize_layout_update(fig=None, n_vis=3):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  for i in range(n_vis*2):
    fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-27, 33],), yaxis = dict(nticks=5, range=[-2, 12],), zaxis = dict(nticks=10, range=[-31, 19],), aspectmode='manual', aspectratio=dict(x=4, y=2, z=3))
  return fig

def make_visualize(input_trajectory_test, output_test_xyz, output_trajectory_test_xyz, output_trajectory_test_startpos, input_trajectory_test_uv, input_trajectory_test_lengths, output_trajectory_test_mask, visualization_path, mae_loss_trajectory, mae_loss_3axis, trajectory_type, animation_visualize_flag, gt_eot, pred_eot, accepted_3axis_maxdist, maxdist_3axis):
  # Visualize by make a subplots of trajectory
  n_vis = 5
  if n_vis > args.batch_size:
    n_vis = args.batch_size
  if n_vis > input_trajectory_test_uv.shape[0]:
    n_vis = input_trajectory_test_uv.shape[0]

  fig = make_subplots(rows=n_vis*2, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}], [{'colspan':2}, None]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  # Random the index the be visualize
  vis_idx = np.random.choice(a=np.arange(input_trajectory_test_uv.shape[0]), size=(n_vis), replace=False)
  # Visualize a trajectory
  fig = visualize_trajectory(input=input_trajectory_test_uv, output=pt.mul(output_test_xyz, output_trajectory_test_mask[..., :-1]), trajectory_gt=output_trajectory_test_xyz[..., :-1], trajectory_startpos=output_trajectory_test_startpos[..., :-1], lengths=input_trajectory_test_lengths, mask=output_trajectory_test_mask[..., :-1], fig=fig, flag='Test', n_vis=n_vis, mae_loss_trajectory=mae_loss_trajectory.cpu().detach().numpy(), mae_loss_3axis=mae_loss_3axis.cpu().detach().numpy(), vis_idx=vis_idx, pred_eot=pred_eot, gt_eot=gt_eot, accepted_3axis_maxdist=accepted_3axis_maxdist.cpu().detach().numpy(), maxdist_3axis=maxdist_3axis.cpu().detach().numpy())
  # Adjust the layout/axis
  # AUTO SCALED/PITCH SCALED
  fig.update_layout(height=2048, width=1500, autosize=True, title="Testing on {} trajectory: Trajectory Visualization with EOT flag(Col1=PITCH SCALED, Col2=AUTO SCALED)".format(trajectory_type))
  fig = visualize_layout_update(fig=fig, n_vis=n_vis)
  plotly.offline.plot(fig, filename='./{}/trajectory_visualization_depth.html'.format(args.visualization_path), auto_open=True)
  if animation_visualize_flag:
    trajectory_animation(output_xyz=pt.mul(output_test_xyz, output_trajectory_test_mask[..., :-1]), gt_xyz=output_trajectory_test_xyz[..., :-1], input_uv=input_trajectory_test_uv, lengths=input_trajectory_test_lengths, mask=output_trajectory_test_mask[..., :-1], n_vis=n_vis, html_savepath=visualization_path, vis_idx=vis_idx)
  input("Continue plotting...")

def visualize_layout_update(fig=None, n_vis=7):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  # fig.update_layout(height=1920, width=1080, margin=dict(l=0, r=0, b=5,t=5,pad=1), autosize=False)
  for i in range(n_vis*2):
    if i%2==0:
      # Set the figure in column 1 (fig0, 2, 4, ...) into a pitch scaled
      # fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-5, 5],), yaxis = dict(nticks=5, range=[-2, 4],), zaxis = dict(nticks=10, range=[-5, 5],),)
      fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-27, 33],), yaxis = dict(nticks=5, range=[-2, 12],), zaxis = dict(nticks=10, range=[-31, 19],), aspectmode='manual', aspectratio=dict(x=4, y=2, z=3))
    fig['layout']['scene{}'.format(i+1)]['camera'].update(projection=dict(type="perspective"))
  return fig

def visualize_trajectory(input, output, trajectory_gt, trajectory_startpos, lengths, mask, mae_loss_trajectory, mae_loss_3axis, vis_idx, gt_eot, pred_eot, accepted_3axis_maxdist, maxdist_3axis, fig=None, flag='test', n_vis=5):
  # marker_dict for contain the marker properties
  marker_dict_gt = dict(color='rgba(0, 0, 255, 0.7)', size=3)
  marker_dict_pred = dict(color='rgba(255, 0, 0, 0.7)', size=3)
  # Visualize the displacement
  marker_dict_u = dict(color='rgba(255, 0, 0, 0.7)', size=4)
  marker_dict_v = dict(color='rgba(0, 255, 0, 0.7)', size=4)
  marker_dict_depth = dict(color='rgba(0, 0, 255, 0.5)', size=4)
  marker_dict_eot = dict(color='rgba(0, 255, 0, 1)', size=5)

  # MAE Loss
  # detach() for visualization
  input = input.cpu().detach().numpy()
  gt_eot = gt_eot.cpu().detach().numpy()
  pred_eot = pred_eot.cpu().detach().numpy()
  output = output.cpu().detach().numpy()
  trajectory_gt = trajectory_gt.cpu().detach().numpy()
  # Iterate to plot each trajectory
  count = 1
  for idx, i in enumerate(vis_idx):
    for col_idx in range(1, 3):
      fig.add_trace(go.Scatter3d(x=output[i][:lengths[i]+1, 0], y=output[i][:lengths[i]+1, 1], z=output[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}, MAE_trajectory = {:.3f}, MaxDist = {}".format(flag, i, TrajectoryLoss(pt.tensor(output[i]).to(device), pt.tensor(trajectory_gt[i]).to(device), mask=mask[i]), mae_loss_trajectory[i], maxdist_3axis[i, :])), row=idx+count, col=col_idx)
      fig.add_trace(go.Scatter3d(x=trajectory_gt[i][:lengths[i]+1, 0], y=trajectory_gt[i][:lengths[i]+1, 1], z=trajectory_gt[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+count, col=col_idx)
    count +=1

  # Iterate to plot each displacement of (u, v, depth)
  for idx, i in enumerate(vis_idx):
    col_idx = 1
    row_idx = (idx*2) + 2
    fig.add_trace(go.Scatter(x=np.arange(gt_eot[i][:lengths[i]].shape[0]), y=gt_eot[i][:lengths[i]], marker=marker_dict_gt, mode='markers+lines', name='{}-Trajectory [{}], EOT'.format(flag, i)), row=row_idx, col=col_idx)
    fig.add_trace(go.Scatter(x=np.arange(pred_eot[i][:lengths[i]].shape[0]), y=pred_eot[i][:lengths[i]], marker=marker_dict_eot, mode='markers+lines', name='{}-Trajectory [{}], EOT'.format(flag, i)), row=row_idx, col=col_idx)
    fig.add_trace(go.Scatter(x=np.arange(input[i][:lengths[i], 0].shape[0]), y=np.diff(input[i][:lengths[i]+1, 0]), marker=marker_dict_gt, mode='lines', name='{}-Trajectory [{}], U'.format(flag, i)), row=row_idx, col=col_idx)
    fig.add_trace(go.Scatter(x=np.arange(input[i][:lengths[i], 1].shape[0]), y=np.diff(input[i][:lengths[i]+1, 1]), marker=marker_dict_gt, mode='lines', name='{}-Trajectory [{}], V'.format(flag, i)), row=row_idx, col=col_idx)

  return fig


def GravityLoss(output, trajectory_gt, mask, lengths):
  # Compute the 2nd finite difference of the y-axis to get the gravity should be equal in every time step
  gravity_constestt_penalize = 0
  # Gaussian blur kernel for not get rid of the input information
  gaussian_blur = pt.tensor([0.25, 0.5, 0.25], dtype=pt.float32).view(1, 1, -1).to(device)
  # Kernel weight for performing a finite difference
  kernel_weight = pt.tensor([-1., 0., 1.], dtype=pt.float32).view(1, 1, -1).to(device)
  # Apply Gaussian blur and finite difference to trajectory_gt
  for i in range(trajectory_gt.shape[0]):
    # print(trajectory_gt[i][:lengths[i]+1, 1])
    # print(trajectory_gt[i][:lengths[i]+1, 1].shape)
    if trajectory_gt[i][:lengths[i]+1, 1].shape[0] < 6:
      print("The trajectory is too shorter to perform a convolution")
      continue
    trajectory_gt_yaxis_1st_gaussian_blur = pt.nn.functional.conv1d(trajectory_gt[i][:lengths[i]+1, 1].view(1, 1, -1), gaussian_blur)
    trajectory_gt_yaxis_1st_finite_difference = pt.nn.functional.conv1d(trajectory_gt_yaxis_1st_gaussian_blur, kernel_weight)
    trajectory_gt_yaxis_2nd_gaussian_blur = pt.nn.functional.conv1d(trajectory_gt_yaxis_1st_finite_difference, gaussian_blur)
    trajectory_gt_yaxis_2nd_finite_difference = pt.nn.functional.conv1d(trajectory_gt_yaxis_2nd_gaussian_blur, kernel_weight)
    # Apply Gaussian blur and finite difference to trajectory_gt
    output_yaxis_1st_gaussian_blur = pt.nn.functional.conv1d(output[i][:lengths[i]+1, 1].view(1, 1, -1), gaussian_blur)
    output_yaxis_1st_finite_difference = pt.nn.functional.conv1d(output_yaxis_1st_gaussian_blur, kernel_weight)
    output_yaxis_2nd_gaussian_blur = pt.nn.functional.conv1d(output_yaxis_1st_finite_difference, gaussian_blur)
    output_yaxis_2nd_finite_difference = pt.nn.functional.conv1d(output_yaxis_2nd_gaussian_blur, kernel_weight)
    # Compute the penalize term
    # print(trajectory_gt_yaxis_2nd_finite_difference, output_yaxis_2nd_finite_difference)
    gravity_constestt_penalize += (pt.sum((trajectory_gt_yaxis_2nd_finite_difference - output_yaxis_2nd_finite_difference)**2))

  return pt.mean(gravity_constestt_penalize)

def TrajectoryLoss(output, trajectory_gt, mask, lengths=None, delmask=True):
  # L2 loss of reconstructed trajectory
  trajectory_loss = (pt.sum((((trajectory_gt - output))**2) * mask) / pt.sum(mask))
  return trajectory_loss

def projectToWorldSpace(screen_space, depth, projection_matrix, camera_to_world_matrix, width, height):
  # print(screen_space.shape, depth.shape)
  depth = depth.view(-1)
  screen_width = width
  screen_height = height
  screen_space = pt.div(screen_space, pt.tensor([screen_width, screen_height]).to(device)) # Normalize : (width, height) -> (-1, 1)
  screen_space = (screen_space * 2.0) - pt.ones(size=(screen_space.size()), dtype=pt.float32).to(device) # Normalize : (width, height) -> (-1, 1)
  screen_space = (screen_space.t() * depth).t()   # Normalize : (-1, 1) -> (-depth, depth) : Camera space (x', y', d, 1)
  screen_space = pt.stack((screen_space[:, 0], screen_space[:, 1], depth, pt.ones(depth.shape[0], dtype=pt.float32).to(device)), axis=1) # Stack the screen with depth and w ===> (x, y, depth, 1)
  screen_space = ((camera_to_world_matrix @ projection_matrix) @ screen_space.t()).t() # Reprojected
  return screen_space[:, :3]

def cumsum_trajectory(output, trajectory, trajectory_startpos):
  '''
  Perform a cummulative summation to the output
  Argument :
  1. output : The displacement from the network with shape = (batch_size, sequence_length,)
  2. trajectory : The input_trajectory (displacement of u, v) with shape = (batch_size, sequence_length, 2)
  3. trajectory_startpos : The start position of input trajectory with shape = (batch_size, 1, )
  Output :
  1. output : concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1,)
  2. trajectory_temp : u, v by concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1, 2)
  '''
  # Apply cummulative summation to output
  # trajectory_temp : concat with startpos and stack back to (batch_size, sequence_length+1, 2)
  trajectory_temp = pt.stack([pt.cat([trajectory_startpos[i][:, :2], trajectory[i].clone().detach()]) for i in range(trajectory_startpos.shape[0])])
  # trajectory_temp : perform cumsum along the sequence_length axis
  trajectory_temp = pt.cumsum(trajectory_temp, dim=1)
  # output : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output = pt.stack([pt.cat([trajectory_startpos[i][:, -1].view(-1, 1), output[i]]) for i in range(trajectory_startpos.shape[0])])
  # output : perform cumsum along the sequence_length axis
  output = pt.cumsum(output, dim=1)
  # print(output.shape, trajectory_temp.shape)
  return output, trajectory_temp

def add_noise(input_trajectory, startpos, lengths):
  factor = np.random.uniform(low=0.6, high=0.95)
  if args.noise_sd is None:
    noise_sd = np.random.uniform(low=0.3, high=0.7)
  else:
    noise_sd = args.noise_sd
  input_trajectory = pt.cat((startpos[..., [0, 1, -1]], input_trajectory), dim=1)
  input_trajectory = pt.cumsum(input_trajectory, dim=1)
  noise_uv = pt.normal(mean=0.0, std=noise_sd, size=input_trajectory[..., :-1].shape).to(device)
  ''' For see the maximum range of noise in uv-coordinate space
  for i in np.arange(0.3, 2, 0.1):
    noise_uv = pt.normal(mean=0.0, std=i, size=input_trajectory[..., :-1].shape).to(device)
    x = []
    for j in range(100):
     x.append(np.all(noise_uv.cpu().numpy() < 3))
    print('{:.3f} : {} with max = {:.3f}, min = {:.3f}'.format(i, np.all(x), pt.max(noise_uv), pt.min(noise_uv)))
  '''
  masking_noise = pt.nn.init.uniform_(pt.empty(input_trajectory[..., :-1].shape)).to(device) > np.random.rand(1)[0]
  n_noise = int(args.batch_size * factor)
  noise_idx = np.random.choice(a=args.batch_size, size=(n_noise,), replace=False)
  input_trajectory[noise_idx, :, :-1] += noise_uv[noise_idx, :, :] * masking_noise[noise_idx, :, :]
  input_trajectory = pt.tensor(np.diff(input_trajectory.cpu().numpy(), axis=1)).to(device)
  return input_trajectory

def evaluateModel(output, trajectory_gt, mask, lengths, threshold=1, delmask=True):
  mae_loss_3axis = pt.sum(((pt.abs(trajectory_gt - output)) * mask), axis=1) / pt.sum(mask, axis=1)
  maxdist_3axis = pt.max(pt.abs(trajectory_gt - output) * mask, dim=1)[0]
  accepted_3axis_maxdist = pt.sum((pt.sum(maxdist_3axis < threshold, axis=1) == 3))
  mae_loss_trajectory = pt.sum(mae_loss_3axis, axis=1) / 3
  print("Accepted 3-Axis(X, Y, Z) Maxdist < {} : {}".format(threshold, accepted_3axis_maxdist))
  accepted_3axis_loss = pt.sum((pt.sum(mae_loss_3axis < threshold, axis=1) == 3))
  print("Accepted 3-Axis(X, Y, Z) loss < {} : {}".format(threshold, accepted_3axis_loss))
  accepted_trajectory_loss = pt.sum(mae_loss_trajectory < threshold)
  print("Accepted trajectory loss < {} : {}".format(threshold, accepted_trajectory_loss))
  return accepted_3axis_maxdist, accepted_3axis_loss, accepted_trajectory_loss, mae_loss_trajectory, mae_loss_3axis, maxdist_3axis

def eot_metrics_log(eot_gt, output_eot, lengths, flag):
  output_eot = output_eot > 0.5
  # Output of confusion_matrix.ravel() = [TN, FP ,FN, TP]
  cm_each_trajectory = np.array([confusion_matrix(y_pred=output_eot[i][:lengths[i], :], y_true=eot_gt[i][:lengths[i]]).ravel() for i in range(lengths.shape[0])])
  n_accepted_trajectory = np.sum(np.logical_and(cm_each_trajectory[:, 1]==0., cm_each_trajectory[:, 2] == 0.))
  cm_batch = np.sum(cm_each_trajectory, axis=0)
  tn, fp, fn, tp = cm_batch
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1_score = 2 * (precision * recall) / (precision + recall)

def EndOfTrajectoryLoss(output_eot, eot_gt, eot_startpos, mask, lengths, flag='Train'):
  # Here we use output mask so we need to append the startpos to the output_eot before multiplied with mask(already included the startpos)
  mask = pt.unsqueeze(mask, dim=-1)
  eot_gt = pt.unsqueeze(eot_gt, dim=-1)
  output_eot *= mask
  eot_gt *= mask

  # Log the precision, recall, confusion_matrix and using wandb
  eot_gt_log = eot_gt.clone().cpu().detach().numpy()
  output_eot_log = output_eot.clone().cpu().detach().numpy()
  eot_metrics_log(eot_gt=eot_gt_log, output_eot=output_eot_log, lengths=lengths.cpu().detach().numpy(), flag=flag)

  # Implement from scratch
  # Flatten and concat all trajectory together
  eot_gt = pt.cat(([eot_gt[i][:lengths[i]+1] for i in range(eot_startpos.shape[0])]))
  output_eot = pt.cat(([output_eot[i][:lengths[i]+1] for i in range(eot_startpos.shape[0])]))
  # Class weight for imbalance class problem
  pos_weight = pt.sum(eot_gt == 0)/pt.sum(eot_gt==1)
  neg_weight = 1
  # Prevent of pt.log(-value)
  eps = 1e-10
  # Calculate the BCE loss
  eot_loss = pt.mean(-((pos_weight * eot_gt * pt.log(output_eot + eps)) + (neg_weight * (1-eot_gt)*pt.log(1-output_eot + eps))))
  return eot_loss

def predict(input_trajectory_test, input_trajectory_test_mask, input_trajectory_test_lengths, input_trajectory_test_startpos, model_eot, model_depth, output_trajectory_test, output_trajectory_test_mask, output_trajectory_test_lengths, output_trajectory_test_startpos, output_trajectory_test_xyz, projection_matrix, camera_to_world_matrix, width, height, threshold, trajectory_type, animation_visualize_flag, visualize_trajectory_flag=True, visualization_path='./visualize_html/'):
  # testing RNN/LSTM model
  # Run over each example
  # test a model

  # Initial the state for model and Discriminator for EOT and Depth
  hidden_G_eot = model_eot.initHidden(batch_size=args.batch_size)
  cell_state_G_eot = model_eot.initCellState(batch_size=args.batch_size)
  hidden_G_depth = model_depth.initHidden(batch_size=args.batch_size)
  cell_state_G_depth = model_depth.initCellState(batch_size=args.batch_size)
  # Initialize the real/fake label for adversarial loss
  real_label = pt.ones(size=(args.batch_size, 1)).to(device)
  fake_label = pt.zeros(size=(args.batch_size, 1)).to(device)

  # Training mode
  model_eot.eval()
  model_depth.eval()
  # Add noise on the fly
  input_trajectory_test_gt = input_trajectory_test.clone()
  if args.noise:
    input_trajectory_test = add_noise(input_trajectory=input_trajectory_test, startpos=input_trajectory_test_startpos, lengths=input_trajectory_test_lengths)

  input_trajectory_test_eot = input_trajectory_test[..., :-1].clone()

  # Forward PASSING
  # Forward pass for training a model
  # Predict the EOT
  output_test_eot, (_, _) = model_eot(input_trajectory_test_eot, hidden_G_eot, cell_state_G_eot, lengths=input_trajectory_test_lengths)
  output_test_eot = pt.sigmoid(output_test_eot).clone()
  input_trajectory_test = pt.cat((input_trajectory_test[..., :-1], output_test_eot), dim=2)
  # Predict the DEPTH
  output_test_depth, (_, _) = model_depth(input_trajectory_test, hidden_G_depth, cell_state_G_depth, lengths=input_trajectory_test_lengths)
  # (This step we get the displacement of depth by input the displacement of u and v)
  # Apply cummulative summation to output using cumsum_trajectory function
  output_test_depth, input_trajectory_test_uv = cumsum_trajectory(output=output_test_depth, trajectory=input_trajectory_test_gt[..., :-1], trajectory_startpos=input_trajectory_test_startpos[..., :-1])
  # Project the (u, v, depth) to world space
  output_test_xyz = pt.stack([projectToWorldSpace(screen_space=input_trajectory_test_uv[i], depth=output_test_depth[i], projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, width=width, height=height) for i in range(output_test_depth.shape[0])])


  ####################################
  ############# EOT&Depth ############
  ####################################
  # model Loss = adversarialLoss + trajectory_loss + gravity_loss + eot_loss
  trajectory_loss = TrajectoryLoss(output=output_test_xyz, trajectory_gt=output_trajectory_test_xyz[..., :-1], mask=output_trajectory_test_mask[..., :-1], lengths=output_trajectory_test_lengths)
  gravity_loss = GravityLoss(output=output_test_xyz, trajectory_gt=output_trajectory_test_xyz[..., :-1], mask=output_trajectory_test_mask[..., :-1], lengths=output_trajectory_test_lengths)
  if args.no_gt_eot:
    eot_loss = pt.tensor(0.).to(device)
  else:
    eot_loss = EndOfTrajectoryLoss(output_eot=output_test_eot, eot_gt=input_trajectory_test_gt[..., -1], mask=input_trajectory_test_mask[..., -1], lengths=input_trajectory_test_lengths, eot_startpos=input_trajectory_test_startpos[..., -1], flag='test')
  test_loss = trajectory_loss + gravity_loss + eot_loss

  ####################################
  ############# Evaluation ###########
  ####################################
  # Calculate loss per trajectory
  accepted_3axis_maxdist, accepted_3axis_loss, accepted_trajectory_loss, mae_loss_trajectory, mae_loss_3axis, maxdist_3axis = evaluateModel(output=output_test_xyz, trajectory_gt=output_trajectory_test_xyz[..., :-1], mask=output_trajectory_test_mask[..., :-1], lengths=output_trajectory_test_lengths, threshold=threshold)

  print('Test Loss : {:.3f}'.format(test_loss.item()), end=', ')
  print('Trajectory Loss : {:.3f}'.format(trajectory_loss.item()), end=', ')
  print('Gravity Loss : {:.3f}'.format(gravity_loss.item()), end=', ')
  print('EndOfTrajectory Loss : {:.3f}'.format(eot_loss.item()))

  if visualize_trajectory_flag == True:
    make_visualize(input_trajectory_test=input_trajectory_test, output_test_xyz=output_test_xyz, output_trajectory_test_xyz=output_trajectory_test_xyz, output_trajectory_test_startpos=output_trajectory_test_startpos, input_trajectory_test_lengths=input_trajectory_test_lengths, input_trajectory_test_uv=input_trajectory_test_uv, output_trajectory_test_mask=output_trajectory_test_mask, visualization_path=visualization_path, mae_loss_trajectory=mae_loss_trajectory, mae_loss_3axis=mae_loss_3axis, trajectory_type=trajectory_type, animation_visualize_flag=animation_visualize_flag, gt_eot=input_trajectory_test_gt[..., -1], pred_eot=input_trajectory_test[..., -1], accepted_3axis_maxdist=accepted_3axis_maxdist, maxdist_3axis=maxdist_3axis)

  return accepted_3axis_maxdist, accepted_3axis_loss, accepted_trajectory_loss

def initialize_folder(path):
  if not os.path.exists(path):
      os.makedirs(path)

def collate_fn_padd(batch):
    '''
    Padding batch of variable length
    '''
    if args.unity:
      input_col = [4, 5, -2]
      input_startpos_col = [4, 5, 6, -2]
      output_col = [6, -2]
      output_startpos_col = [0, 1, 2, -2]
      output_xyz_col = [0, 1, 2, -2]
    elif args.mocap:
      input_col = [3, 4, -1]
      input_startpos_col = [3, 4, 5, -1]
      output_col = [5, -1]
      output_startpos_col = [0, 1, 2, -1]
      output_xyz_col = [0, 1, 2, -1]
    elif args.predicted_eot:
      input_col = [3, 4, -1]
      input_startpos_col = [3, 4, 5, -1]
      output_col = [5, -1]
      output_startpos_col = [0, 1, 2, -1]
      output_xyz_col = [0, 1, 2, -1]
    else :
      print("Please Specify the column convention")
      exit()

    padding_value = -10
    ## Get sequence lengths
    lengths = pt.tensor([trajectory[1:, :].shape[0] for trajectory in batch])
    # Input features : columns 4-5 contain u, v in screen space
    ## Padding 
    input_batch = [pt.Tensor(trajectory[1:, input_col]) for trajectory in batch] # Mocap : Manually Labeled
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=padding_value)
    ## Retrieve initial position (u, v, depth)
    input_startpos = pt.stack([pt.Tensor(trajectory[0, input_startpos_col]) for trajectory in batch]) # Mocap : Manually Labeled
    input_startpos = pt.unsqueeze(input_startpos, dim=1)
    ## Compute mask
    input_mask = (input_batch != padding_value)

    # Output features : columns 6 cotain depth from camera to projected screen
    ## Padding
    output_batch = [pt.Tensor(trajectory[:, output_col]) for trajectory in batch] # Mocap
    output_batch = pad_sequence(output_batch, batch_first=True)
    ## Retrieve initial position
    output_startpos = pt.stack([pt.Tensor(trajectory[0, output_startpos_col]) for trajectory in batch]) # Mocap : Manually Labeled
    output_startpos = pt.unsqueeze(output_startpos, dim=1)
    ## Retrieve the x, y, z in world space for compute the reprojection error (x', y', z' <===> x, y, z)
    # output_xyz = [pt.Tensor(trajectory[:, [0, 1, 2, -2]]) for trajectory in batch]
    output_xyz = [pt.Tensor(trajectory[:, output_xyz_col]) for trajectory in batch] # Mocap :  Manually Labeled
    output_xyz = pad_sequence(output_xyz, batch_first=True, padding_value=padding_value)
    ## Compute mask
    output_mask = (output_xyz != padding_value)
    ## Compute cummulative summation to form a trajectory from displacement every columns except the end_of_trajectory
    # print(output_xyz[..., :-1].shape, pt.unsqueeze(output_xyz[..., -1], dim=2).shape)
    output_xyz = pt.cat((pt.cumsum(output_xyz[..., :-1], dim=1), pt.unsqueeze(output_xyz[..., -1], dim=2)), dim=2)

    # print("Mask shape : ", mask.shape)
    return {'input':[input_batch, lengths, input_mask, input_startpos],
            'output':[output_batch, lengths+1, output_mask, output_startpos, output_xyz]}

def get_model(input_size, output_size, model_arch):
  if model_arch=='bigru_residual_add':
    model_eot = BiGRUResidualAdd(input_size=2, output_size=1)
    model_depth = BiGRUResidualAdd(input_size=3, output_size=1)
  else :
    print("Please input correct model architecture : gru, bigru, lstm, bilstm")
    exit()

  return model_eot, model_depth

def load_checkpoint(model_eot, model_depth):
  if os.path.isfile(args.load_checkpoint):
    print("[#] Found the checkpoint...")
    checkpoint = pt.load(args.load_checkpoint, map_location='cuda:0')
    # Load optimizer, learning rate, decay and scheduler parameters
    model_eot.load_state_dict(checkpoint['model_eot'])
    model_depth.load_state_dict(checkpoint['model_depth'])
    start_epoch = checkpoint['epoch']
    return model_eot, model_depth, start_epoch

  else:
    print("[#] Checkpoint not found...")
    exit()

if __name__ == '__main__':
  print('[#]testing : Trajectory Estimation')
  # Argumentparser for input
  parser = argparse.ArgumentParser(description='Predict the 3D projectile')
  parser.add_argument('--dataset_test_path', dest='dataset_test_path', type=str, help='Path to testing set', required=True)
  parser.add_argument('--batch_size', dest='batch_size', type=int, help='Size of batch', default=50)
  parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, help='Type of trajectory(Rolling, Projectile, MagnusProjectile)', default='Projectile')
  parser.add_argument('--no_visualize', dest='visualize_trajectory_flag', help='No Visualize the trajectory', action='store_false')
  parser.add_argument('--visualize', dest='visualize_trajectory_flag', help='Visualize the trajectory', action='store_true')
  parser.add_argument('--load_checkpoint', dest='load_checkpoint', type=str, help='Path to load a tested model checkpoint', default=None)
  parser.add_argument('--visualization_path', dest='visualization_path', type=str, help='Path to visualization directory', default='./visualize_html/')
  parser.add_argument('--cam_params_file', dest='cam_params_file', type=str, help='Path to camera parameters file(Intrinsic/Extrinsic)')
  parser.add_argument('--cuda_device_num', dest='cuda_device_num', type=int, help='Provide cuda device number', default=0)
  parser.add_argument('--model_arch', dest='model_arch', type=str, help='Input the model architecture(lstm, bilstm, gru, bigru)', required=True)
  parser.add_argument('--threshold', dest='threshold', type=float, help='Provide the error threshold of reconstructed trajectory', default=0.8)
  parser.add_argument('--no_animation', dest='animation_visualize_flag', help='Animated visualize flag', action='store_false')
  parser.add_argument('--animation', dest='animation_visualize_flag', help='Animated visualize flag', action='store_true')
  parser.add_argument('--noise', dest='noise', help='Noise on the fly', action='store_true')
  parser.add_argument('--no_noise', dest='noise', help='Noise on the fly', action='store_false')
  parser.add_argument('--noise_sd', dest='noise_sd', help='Std. of noise', type=float, default=None)
  parser.add_argument('--unity', dest='unity', help='Use unity column convention', action='store_true', default=False)
  parser.add_argument('--mocap', dest='mocap', help='Use mocap column convention', action='store_true', default=False)
  parser.add_argument('--predicted_eot', dest='predicted_eot', help='Use predicted_eot column convention', action='store_true', default=False)
  parser.add_argument('--no_gt_eot', dest='no_gt_eot', help='Use predicted_eot column convention', action='store_true', default=False)
  args = parser.parse_args()

  # Initialize folder
  initialize_folder(args.visualization_path)

  # GPU initialization
  if pt.cuda.is_available():
    pt.cuda.set_device(args.cuda_device_num)
    device = pt.device('cuda')
    print('[%]GPU Enabled')
  else:
    device = pt.device('cpu')
    print('[%]GPU Disabled, CPU Enabled')

  # Load camera parameters : ProjectionMatrix and WorldToCameraMatrix
  with open(args.cam_params_file) as cam_params_json:
    cam_params_file = json.load(cam_params_json)
    cam_params = dict({'projectionMatrix':cam_params_file['mainCameraParams']['projectionMatrix'], 'worldToCameraMatrix':cam_params_file['mainCameraParams']['worldToCameraMatrix'], 'width':cam_params_file['mainCameraParams']['width'], 'height':cam_params_file['mainCameraParams']['height']})
  projection_matrix = np.array(cam_params['projectionMatrix']).reshape(4, 4)
  projection_matrix = pt.tensor([projection_matrix[0, :], projection_matrix[1, :], projection_matrix[3, :], [0, 0, 0, 1]], dtype=pt.float32)
  projection_matrix = pt.inverse(projection_matrix).to(device)
  camera_to_world_matrix = pt.inverse(pt.tensor(cam_params['worldToCameraMatrix']).view(4, 4)).to(device)
  width = cam_params['width']
  height = cam_params['height']

  # Create Datasetloader for test and testidation
  print(args.dataset_test_path)
  trajectory_test_dataset = TrajectoryDataset(dataset_path=args.dataset_test_path, trajectory_type=args.trajectory_type)
  trajectory_test_dataloader = DataLoader(trajectory_test_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True, collate_fn=collate_fn_padd, pin_memory=True, drop_last=True)
  # Create Datasetloader for testidation
  trajectory_test_dataset = TrajectoryDataset(dataset_path=args.dataset_test_path, trajectory_type=args.trajectory_type)
  trajectory_test_dataloader = DataLoader(trajectory_test_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True, collate_fn=collate_fn_padd, pin_memory=True, drop_last=True)
  # Cast it to iterable object
  trajectory_test_iterloader = iter(trajectory_test_dataloader)

  # Dataset format
  # Trajectory path : (x0, y0) ... (xn, yn)
  print("======================================================Summary Batch (batch_size = {})=========================================================================".format(args.batch_size))
  for key, batch in enumerate(trajectory_test_dataloader):
    print("Input batch [{}] : batch={}, lengths={}, mask={}, initial position={}".format(key, batch['input'][0].shape, batch['input'][1].shape, batch['input'][2].shape, batch['input'][3].shape))
    print("Output batch [{}] : batch={}, lengths={}, mask={}, initial position={}".format(key, batch['output'][0].shape, batch['output'][1].shape, batch['output'][2].shape, batch['output'][3].shape))
    # Test RNN/LSTM Step
    # 1.Pack the padded
    packed = pt.nn.utils.rnn.pack_padded_sequence(batch['input'][0], batch_first=True, lengths=batch['input'][1], enforce_sorted=False)
    # 2.RNN/LSTM model
    # 3.Unpack the packed
    unpacked = pt.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, padding_value=-10)
    print("Unpacked equality : ", pt.eq(batch['input'][0], unpacked[0]).all())
    print("===============================================================================================================================================================")

  # Model definition
  n_output = 1 # Contain the depth information of the trajectory
  n_input = 3 # Contain following this trajectory parameters (u, v, end_of_trajectory) position from tracking
  model_eot, model_depth = get_model(input_size=n_input, output_size=n_output, model_arch=args.model_arch)
  model_eot = model_eot.to(device)
  model_depth = model_depth.to(device)

  # Load the checkpoint if it's available.
  if args.load_checkpoint is None:
    # Create a model
    print('===>No model checkpoint')
    print('[#] Define the Learning rate, Optimizer, Decay rate and Scheduler...')
  else:
    print('===>Load checkpoint with Optimizer state, Decay and Scheduler state')
    print('[#] Loading ... {}'.format(args.load_checkpoint))
    model_eot, model_depth, start_epoch = load_checkpoint(model_eot, model_depth)

  print('[#]Model Architecture')
  print('####### Model - EOT #######')
  print(model_eot)
  print('####### Model - Depth #######')
  print(model_depth)

  # Test a model iterate over dataloader to get each batch and pass to predict function
  n_accepted_3axis_loss = 0
  n_accepted_trajectory_loss = 0
  n_accepted_3axis_maxdist = 0
  n_trajectory = 0
  for batch_idx, batch_test in enumerate(trajectory_test_dataloader):
    print("[#]Batch-{}".format(batch_idx))
    # Testing set (Each index in batch_test came from the collate_fn_padd)
    input_trajectory_test = batch_test['input'][0].to(device)
    input_trajectory_test_lengths = batch_test['input'][1].to(device)
    input_trajectory_test_mask = batch_test['input'][2].to(device)
    input_trajectory_test_startpos = batch_test['input'][3].to(device)
    output_trajectory_test = batch_test['output'][0].to(device)
    output_trajectory_test_lengths = batch_test['output'][1].to(device)
    output_trajectory_test_mask = batch_test['output'][2].to(device)
    output_trajectory_test_startpos = batch_test['output'][3].to(device)
    output_trajectory_test_xyz = batch_test['output'][4].to(device)

      # Call function to test
    accepted_3axis_maxdist, accepted_3axis_loss, accepted_trajectory_loss = predict(input_trajectory_test=input_trajectory_test, input_trajectory_test_mask = input_trajectory_test_mask,
                                                                 input_trajectory_test_lengths=input_trajectory_test_lengths, input_trajectory_test_startpos=input_trajectory_test_startpos,
                                                                 output_trajectory_test=output_trajectory_test, output_trajectory_test_mask=output_trajectory_test_mask,
                                                                 output_trajectory_test_lengths=output_trajectory_test_lengths, output_trajectory_test_startpos=output_trajectory_test_startpos, output_trajectory_test_xyz=output_trajectory_test_xyz,
                                                                 model_eot=model_eot, model_depth=model_depth,
                                                                 visualize_trajectory_flag=args.visualize_trajectory_flag, projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, threshold=args.threshold, trajectory_type=args.trajectory_type, animation_visualize_flag=args.animation_visualize_flag,
                                                                 width=width, height=height)

    n_accepted_3axis_loss += accepted_3axis_loss
    n_accepted_trajectory_loss += accepted_trajectory_loss
    n_accepted_3axis_maxdist += accepted_3axis_maxdist
    n_trajectory += input_trajectory_test.shape[0]


  print("="*100)
  print("[#]Summary")
  print("Accepted trajectory by MAE Loss : {} from {}".format(n_accepted_trajectory_loss, n_trajectory))
  print("Accepted trajectory by 3axis MAE Loss : {} from {}".format(n_accepted_3axis_loss, n_trajectory))
  print("Accepted trajectory by 3axis MaxDist : {} from {}".format(n_accepted_3axis_maxdist, n_trajectory))
  print("="*100)

  print("[#] Done")
