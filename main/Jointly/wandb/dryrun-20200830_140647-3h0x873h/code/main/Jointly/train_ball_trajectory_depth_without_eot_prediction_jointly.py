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
import PIL.Image
import io
from sklearn.metrics import confusion_matrix
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
from models.Adversarial.bigru_model_residual_generator import Generator
from models.Adversarial.bigru_model_residual_discriminator import Discriminator
from torch.utils.tensorboard import SummaryWriter

def visualize_layout_update(fig=None, n_vis=3):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  for i in range(n_vis*2):
    fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-27, 33],), yaxis = dict(nticks=5, range=[-2, 12],), zaxis = dict(nticks=10, range=[-31, 19],), aspectmode='manual', aspectratio=dict(x=4, y=2, z=3))
  return fig

def make_visualize(input_trajectory_train, output_train_depth, input_trajectory_val, output_val_depth, output_train_xyz, output_trajectory_train_xyz, output_trajectory_train_startpos, input_trajectory_train_lengths, output_trajectory_train_maks, output_val_xyz, output_trajectory_val_xyz, output_trajectory_val_startpos, input_trajectory_val_lengths, output_trajectory_val_mask, visualization_path):
  # Visualize by make a subplots of trajectory
  n_vis = 5
  # Random the index the be visualize
  train_vis_idx = np.random.randint(low=0, high=input_trajectory_train_startpos.shape[0], size=(n_vis))
  val_vis_idx = np.random.randint(low=0, high=input_trajectory_val_startpos.shape[0], size=(n_vis))

  # Visualize the displacement
  fig_displacement = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  visualize_displacement(in_f=input_trajectory_train, out_f=output_train_depth, mask=input_trajectory_train_mask, lengths=input_trajectory_train_lengths, n_vis=n_vis, vis_idx=train_vis_idx, fig=fig_displacement, flag='Train')
  visualize_displacement(in_f=input_trajectory_val, out_f=output_val_depth, mask=input_trajectory_val_mask, lengths=input_trajectory_val_lengths, n_vis=n_vis, vis_idx=val_vis_idx, fig=fig_displacement, flag='Validation')
  wandb.log({"DISPLACEMENT VISUALIZATION":fig_displacement})

  # Visualize the trajectory
  fig_traj = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  # Append the start position and apply cummulative summation for transfer the displacement to the x, y, z coordinate. These will done by visualize_trajectory function
  # Can use mask directly because the mask obtain from full trajectory(Not remove the start pos)
  visualize_trajectory(output=pt.mul(output_train_xyz, output_trajectory_train_mask[..., :-1]), trajectory_gt=output_trajectory_train_xyz[..., :-1], trajectory_startpos=output_trajectory_train_startpos[..., :-1], lengths=input_trajectory_train_lengths, mask=output_trajectory_train_mask[..., :-1], fig=fig_traj, flag='Train', n_vis=n_vis, vis_idx=train_vis_idx)
  visualize_trajectory(output=pt.mul(output_val_xyz, output_trajectory_val_mask[..., :-1]), trajectory_gt=output_trajectory_val_xyz[..., :-1], trajectory_startpos=output_trajectory_val_startpos[..., :-1], lengths=input_trajectory_val_lengths, mask=output_trajectory_val_mask[..., :-1], fig=fig_traj, flag='Validation', n_vis=n_vis, vis_idx=val_vis_idx)
  # Adjust the layout/axis
  # For an AUTO SCALED
  fig_traj.update_layout(height=1920, width=1500, autosize=True)
  # plotly.offline.plot(fig_traj, filename='/{}/trajectory_visualization_depth_auto_scaled.html'.format(args.visualization_path), auto_open=False)
  # wandb.log({"AUTO SCALED : Trajectory Visualization(Col1=Train, Col2=Val)":wandb.Html(open('/{}/trajectory_visualization_depth_auto_scaled.html'.format(args.visualization_path)))})
  # For a PITCH SCALED
  fig = visualize_layout_update(fig=fig_traj, n_vis=n_vis)
  plotly.offline.plot(fig, filename='./{}/trajectory_visualization_depth_pitch_scaled.html'.format(args.visualization_path), auto_open=False)
  wandb.log({"PITCH SCALED : Trajectory Visualization(Col1=Train, Col2=Val)":wandb.Html(open('./{}/trajectory_visualization_depth_pitch_scaled.html'.format(args.visualization_path)))})

  # Visualize the End of trajectory(EOT) flag
  fig_eot = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  visualize_eot(output_eot=input_trajectory_train[..., -1].clone(), eot_gt=output_trajectory_train_xyz[..., -1], eot_startpos=output_trajectory_train_startpos[..., -1], lengths=output_trajectory_train_lengths, mask=output_trajectory_train_mask[..., -1], fig=fig_eot, flag='Train', n_vis=n_vis, vis_idx=train_vis_idx)
  visualize_eot(output_eot=input_trajectory_train[..., -1].clone(), eot_gt=output_trajectory_val_xyz[..., -1], eot_startpos=output_trajectory_val_startpos[..., -1], lengths=output_trajectory_val_lengths, mask=output_trajectory_val_mask[..., -1], fig=fig_eot, flag='Validation', n_vis=n_vis, vis_idx=val_vis_idx)
  wandb.log({"End Of Trajectory flag Prediction : (Col1=Train, Col2=Val)":fig_eot})

def visualize_displacement(in_f, out_f, mask, lengths, vis_idx, fig=None, flag='train', n_vis=5):
  in_f = in_f.cpu().detach().numpy()
  out_f = np.diff(out_f.cpu().detach().numpy(), axis=1)
  lengths = lengths.cpu().detach().numpy()
  marker_dict_gt = dict(color='rgba(0, 0, 255, 0.7)', size=3)
  marker_dict_pred = dict(color='rgba(255, 0, 0, 0.7)', size=3)
  marker_dict_eot = dict(color='rgba(0, 255, 0, 0.7)', size=3)
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=out_f[i][:lengths[i]+1, 0], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-Displacement of DEPTH'.format(flag, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=in_f[i][:lengths[i]+1, 0], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Displacement of U'.format(flag, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=in_f[i][:lengths[i]+1, 1], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Displacement of V'.format(flag, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=in_f[i][:lengths[i]+1, 2], mode='markers+lines', marker=marker_dict_eot, name='{}-traj#{}-Displacement of EOT'.format(flag, i)), row=idx+1, col=col)

def visualize_trajectory(output, trajectory_gt, trajectory_startpos, lengths, mask, vis_idx, fig=None, flag='train', n_vis=5):
  print(output.shape, trajectory_gt.shape)
  # marker_dict for contain the marker properties
  marker_dict_gt = dict(color='rgba(0, 0, 255, 0.2)', size=3)
  marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=3)
  # detach() for visualization
  output = output.cpu().detach().numpy()
  trajectory_gt = trajectory_gt.cpu().detach().numpy()
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter3d(x=output[i][:lengths[i]+1, 0], y=output[i][:lengths[i]+1, 1], z=output[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}".format(flag, i, TrajectoryLoss(pt.tensor(output[i]).to(device), pt.tensor(trajectory_gt[i]).to(device), mask=mask[i]))), row=idx+1, col=col)
    fig.add_trace(go.Scatter3d(x=trajectory_gt[i][:lengths[i]+1, 0], y=trajectory_gt[i][:lengths[i]+1, 1], z=trajectory_gt[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+1, col=col)

def visualize_eot(output_eot, eot_gt, eot_startpos, lengths, mask, vis_idx, fig=None, flag='train', n_vis=5):
  # Add the feature dimension using unsqueeze
  eot_gt = pt.unsqueeze(eot_gt, dim=2)
  mask = pt.unsqueeze(mask, dim=2)
  eot_startpos = pt.unsqueeze(eot_startpos, dim=2)
  # output_eot : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output_eot = pt.stack([pt.cat([eot_startpos[i], output_eot[i]]) for i in range(eot_startpos.shape[0])])
  # Here we use output mask so we need to append the startpos to the output_eot before multiplied with mask(already included the startpos)
  output_eot *= mask
  eot_gt *= mask
  # sigmoid the output to keep each trajectory seperately
  output_eot = pt.sigmoid(output_eot)
  # Weight of positive/negative classes for imbalanced class
  pos_weight = pt.sum(eot_gt == 0)/pt.sum(eot_gt==1)
  neg_weight = 1
  eps = 1e-10
  # Calculate the EOT loss for each trajectory
  eot_loss = pt.mean(-((pos_weight * eot_gt * pt.log(output_eot+eps)) + (neg_weight * (1-eot_gt)*pt.log(1-output_eot+eps))), dim=1).cpu().detach().numpy()

  # detach() for visualization
  output_eot = output_eot.cpu().detach().numpy()
  eot_gt = eot_gt.cpu().detach().numpy()
  lengths = lengths.cpu().detach().numpy()
  # marker_dict for contain the marker properties
  marker_dict_gt = dict(color='rgba(0, 0, 255, .7)', size=5)
  marker_dict_pred = dict(color='rgba(255, 0, 0, .7)', size=5)
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    # print(output_eot[i][:lengths[i]+1, :])
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]).reshape(-1,), y=output_eot[i][:lengths[i], :].reshape(-1,), mode='markers+lines', marker=marker_dict_pred, name="{}-EOT Predicted [{}], EOTLoss = {:.3f}".format(flag, i, eot_loss[i][0])), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]).reshape(-1,), y=eot_gt[i][:lengths[i], :].reshape(-1,), mode='markers+lines', marker=marker_dict_gt, name="{}-Ground Truth EOT [{}]".format(flag, i)), row=idx+1, col=col)

def GravityLoss(output, trajectory_gt, mask, lengths):
  # Compute the 2nd finite difference of the y-axis to get the gravity should be equal in every time step
  gravity_constraint_penalize = 0
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
    gravity_constraint_penalize += (pt.sum((trajectory_gt_yaxis_2nd_finite_difference - output_yaxis_2nd_finite_difference)**2))

  return pt.mean(gravity_constraint_penalize)

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

def adversarialLoss(hidden_D, cell_state_D, real_label, fake_label, discriminator, fake_traj, real_traj, lengths, flag):
  # Adversarial loss choosen by a flag
  if args.adversarial_lossfn == 'gan':
    adversarial_loss = GanLoss(hidden_D, cell_state_D, real_label, fake_label, discriminator, fake_traj, real_traj, lengths, flag)
  elif args.adversarial_lossfn == 'lsgan':
    adversarial_loss = LSGanLoss(hidden_D, cell_state_D, real_label, fake_label, discriminator, fake_traj, real_traj, lengths, flag)
  elif args.adversarial_lossfn == 'wgan':
    adversarial_loss = WGanLoss(hidden_D, cell_state_D, real_label, fake_label, discriminator, fake_traj, real_traj, lengths, flag)
  elif args.adversarial_lossfn == 'wgangp':
    adversarial_loss = WGanGPLoss(hidden_D, cell_state_D, real_label, fake_label, discriminator, fake_traj, real_traj, lengths, flag)
  return adversarial_loss

def GanLoss(hidden_D, cell_state_D, real_label, fake_label, discriminator, fake_traj, real_traj, lengths, flag):
  # Vanilla GAN Loss is the BCELoss for binary classification (1=Real, 0=Fake)
  eps = 1e-10
  sigmoid_layer = pt.nn.Sigmoid()
  # Calculate the BCE loss
  if flag == 'generator':
    pred_label = sigmoid_layer(discriminator(fake_traj, hidden_D, cell_state_D, lengths)[0])
    gan_loss = pt.mean(-((real_label * pt.log(pred_label + eps)) + ((1-real_label)*pt.log(1-pred_label + eps))))
  else:
    pred_fake_label = sigmoid_layer(discriminator(fake_traj, hidden_D, cell_state_D, lengths)[0])
    fake_loss = pt.mean(-((fake_label * pt.log(pred_fake_label + eps)) + ((1-fake_label)*pt.log(1-pred_fake_label + eps))))
    pred_real_label = sigmoid_layer(discriminator(real_traj, hidden_D, cell_state_D, lengths)[0])
    real_loss = pt.mean(-((real_label * pt.log(pred_real_label + eps)) + ((1-real_label)*pt.log(1-pred_real_label + eps))))
    gan_loss = 0.5 * (fake_loss + real_loss)
  return gan_loss

def LSGanLoss(hidden_D, cell_state_D, real_label, fake_label, discriminator, fake_traj, real_traj, lengths, flag):
  # Use the MSELoss instead of BCELoss 
  if flag == 'generator':
    pred_label = discriminator(fake_traj, hidden_D, cell_state_D, lengths)[0]
    lsgan_loss = pt.mean((real_label - pred_label)**2)
  else:
    pred_fake_label = discriminator(fake_traj, hidden_D, cell_state_D, lengths)[0]
    fake_loss = pt.mean((fake_label - pred_fake_label)**2)
    pred_real_label = discriminator(real_traj, hidden_D, cell_state_D, lengths)[0]
    real_loss = pt.mean((real_label - pred_real_label)**2)
    lsgan_loss = 0.5*(fake_loss + real_loss)
  return lsgan_loss

def WGanLoss(hidden_D, cell_state_D, real_label, fake_label, discriminator, fake_traj, real_traj, lengths, flag):
  if flag == 'generator':
    pred_label = discriminator(fake_traj, hidden_D, cell_state_D, lengths)[0]
    wgan_loss = -pt.mean(pred_label)
  else:
    pred_fake_label = discriminator(fake_traj, hidden_D, cell_state_D, lengths)[0]
    pred_real_label = discriminator(real_traj, hidden_D, cell_state_D, lengths)[0]
    wgan_loss = -pt.mean(pred_real_label) + pt.mean(pred_fake_label)
  return wgan_loss

def WGanGPLoss(hidden_D, cell_state_D, real_label, fake_label, discriminator, fake_traj, real_traj, lengths, flag):
  if flag == 'generator':
    pred_label = discriminator(fake_traj, hidden_D, cell_state_D, lengths)[0]
    wgangp_loss = -pt.mean(pred_label)
  else:
    lambda_gp = 10
    gradient_penalty = compute_gradient_penalty(discriminator, hidden_D, cell_state_D, real_traj, fake_traj, lengths)
    pred_fake_label = discriminator(fake_traj, hidden_D, cell_state_D, lengths)[0]
    pred_real_label = discriminator(real_traj, hidden_D, cell_state_D, lengths)[0]
    wgangp_loss = -pt.mean(pred_real_label) + pt.mean(pred_fake_label) + lambda_gp * gradient_penalty
  return wgangp_loss

def compute_gradient_penalty(discriminator, hidden_D, cell_state_D, real_traj, fake_traj, lengths):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = pt.Tensor(np.random.random((real_traj.shape))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_traj + ((1 - alpha) * fake_traj)).requires_grad_(True)
    d_interpolates = discriminator(interpolates, hidden_D, cell_state_D, lengths)[0].to(device)
    fake = pt.autograd.Variable(pt.Tensor(real_traj.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = pt.autograd.grad(
                   outputs = d_interpolates,
                   inputs = interpolates,
                   grad_outputs = fake,
                   create_graph = True,
                   retain_graph = True,
                   only_inputs = True,)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

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
  wandb.log({'{} Precision'.format(flag):precision, '{} Recall'.format(flag):recall, '{} F1-score'.format(flag):f1_score, '{}-#N accepted trajectory(Perfect EOT without FN, FP)'.format(flag):n_accepted_trajectory})

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

def train(output_trajectory_train, output_trajectory_train_mask, output_trajectory_train_lengths, output_trajectory_train_startpos, output_trajectory_train_xyz, input_trajectory_train, input_trajectory_train_mask, input_trajectory_train_lengths, input_trajectory_train_startpos, generator_eot, generator_depth, discriminator, output_trajectory_val, output_trajectory_val_mask, output_trajectory_val_lengths, output_trajectory_val_startpos, output_trajectory_val_xyz, input_trajectory_val, input_trajectory_val_mask, input_trajectory_val_lengths, input_trajectory_val_startpos, projection_matrix, camera_to_world_matrix, epoch, n_epochs, vis_signal, optimizer_D, optimizer_G, width, height, train_generator, train_discriminator, visualize_trajectory_flag=True, visualization_path='./visualize_html/'):
  # Training RNN/LSTM model
  # Run over each example
  # Train a model
  # Initial the state for Generator and Discriminator for EOT and Depth
  hidden_G_eot = generator_eot.initHidden(batch_size=args.batch_size)
  cell_state_G_eot = generator_eot.initCellState(batch_size=args.batch_size)
  hidden_G_depth = generator_depth.initHidden(batch_size=args.batch_size)
  cell_state_G_depth = generator_depth.initCellState(batch_size=args.batch_size)
  hidden_D = discriminator.initHidden(batch_size=args.batch_size)
  cell_state_D = discriminator.initCellState(batch_size=args.batch_size)
  # Initialize the real/fake label for adversarial loss
  real_label = pt.ones(size=(args.batch_size, 1)).to(device)
  fake_label = pt.zeros(size=(args.batch_size, 1)).to(device)

  # Training mode
  generator_eot.train()
  generator_depth.train()
  # Add noise on the fly
  input_trajectory_train_gt = input_trajectory_train.clone()
  input_trajectory_val_gt = input_trajectory_val.clone()
  if args.noise:
    input_trajectory_train = add_noise(input_trajectory=input_trajectory_train, startpos=input_trajectory_train_startpos, lengths=input_trajectory_train_lengths)
    input_trajectory_train_eot = input_trajectory_train[..., :-1].clone()
    input_trajectory_val = add_noise(input_trajectory=input_trajectory_val, startpos=input_trajectory_val_startpos, lengths=input_trajectory_val_lengths)
    input_trajectory_val_eot = input_trajectory_val[..., :-1].clone()

  # Forward PASSING
  # Forward pass for training a model
  # Predict the EOT
  output_train_eot, (_, _) = generator_eot(input_trajectory_train_eot, hidden_G_eot, cell_state_G_eot, lengths=input_trajectory_train_lengths)
  output_train_eot = pt.sigmoid(output_train_eot).clone()
  input_trajectory_train = pt.cat((input_trajectory_train[..., :-1], output_train_eot), dim=2)
  # Predict the DEPTH
  output_train_depth, (_, _) = generator_depth(input_trajectory_train, hidden_G_depth, cell_state_G_depth, lengths=input_trajectory_train_lengths)
  # (This step we get the displacement of depth by input the displacement of u and v)
  # Apply cummulative summation to output using cumsum_trajectory function
  output_train_depth, input_trajectory_train_uv = cumsum_trajectory(output=output_train_depth, trajectory=input_trajectory_train_gt[..., :-1], trajectory_startpos=input_trajectory_train_startpos[..., :-1])
  # Project the (u, v, depth) to world space
  output_train_xyz = pt.stack([projectToWorldSpace(screen_space=input_trajectory_train_uv[i], depth=output_train_depth[i], projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, width=width, height=height) for i in range(output_train_depth.shape[0])])

  ####################################
  ############# Generator ############
  ####################################
  optimizer_G.zero_grad() # Clear existing gradients from previous epoch
  # Generator Loss = adversarialLoss + trajectory_loss + gravity_loss
  adversarial_loss_G = adversarialLoss(hidden_D = hidden_D, cell_state_D = cell_state_D, real_label=real_label, fake_label=fake_label, discriminator=discriminator, real_traj=output_trajectory_train_xyz[..., :-1], fake_traj=output_train_xyz, lengths=output_trajectory_train_lengths, flag='generator')
  trajectory_loss = TrajectoryLoss(output=output_train_xyz, trajectory_gt=output_trajectory_train_xyz[..., :-1], mask=output_trajectory_train_mask[..., :-1], lengths=output_trajectory_train_lengths)
  gravity_loss = GravityLoss(output=output_train_xyz, trajectory_gt=output_trajectory_train_xyz[..., :-1], mask=output_trajectory_train_mask[..., :-1], lengths=output_trajectory_train_lengths)
  eot_loss = EndOfTrajectoryLoss(output_eot=output_train_eot, eot_gt=input_trajectory_train_gt[..., -1], mask=input_trajectory_train_mask[..., -1], lengths=input_trajectory_train_lengths, eot_startpos=input_trajectory_train_startpos[..., -1], flag='Train')
  # Sum up all generator loss 
  loss_G = adversarial_loss_G + trajectory_loss + gravity_loss + eot_loss
  if train_generator:
    generator.train()
    loss_G.backward()
    for p in generator_eot.parameters():
        p.data.clamp_(-args.clip, args.clip)
    for p in generator_depth.parameters():
        p.data.clamp_(-args.clip, args.clip)
    optimizer_G.step()


  ####################################
  ########### Discriminator ##########
  ####################################
  optimizer_D.zero_grad() # Clear existing gradients from previous epoch
  # Fake = Reconstructed(Generated)
  # Real = Ground truth 
  adversarial_loss_D = adversarialLoss(hidden_D = hidden_D, cell_state_D = cell_state_D, real_label=real_label, fake_label=fake_label, discriminator=discriminator, real_traj=output_trajectory_train_xyz[..., :-1], fake_traj=output_train_xyz, lengths=output_trajectory_train_lengths, flag='discriminator')
  loss_D = adversarial_loss_D

  if train_discriminator:
    discriminator.train()
    loss_D.backward()
    # Clip weights of discriminator
    for p in discriminator.parameters():
        p.data.clamp_(-args.clip, args.clip)
    optimizer_D.step()

  # Evaluating mode
  generator_eot.eval()
  generator_depth.eval()
  # Forward pass for validate a model
  output_val_eot, (_, _) = generator_eot(input_trajectory_val_eot, hidden_G_eot, cell_state_G_eot, lengths=input_trajectory_val_lengths)
  output_val_eot = pt.sigmoid(output_val_eot).clone()
  input_trajectory_val = pt.cat((input_trajectory_val[..., :-1], output_val_eot), dim=2)
  # Predict the DEPTH
  output_val_depth, (_, _) = generator_depth(input_trajectory_val, hidden_G_depth, cell_state_G_depth, lengths=input_trajectory_val_lengths)
  # (This step we get the displacement of depth by input the displacement of u and v)
  # Apply cummulative summation to output using cumsum_trajectory function
  output_val_depth, input_trajectory_val_uv = cumsum_trajectory(output=output_val_depth, trajectory=input_trajectory_val_gt[..., :-1], trajectory_startpos=input_trajectory_val_startpos[..., :-1])
  # Project the (u, v, depth) to world space
  output_val_xyz = pt.stack([projectToWorldSpace(screen_space=input_trajectory_val_uv[i], depth=output_val_depth[i], projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, width=width, height=height) for i in range(output_val_depth.shape[0])])

  # Calculate loss of unprojected trajectory
  val_loss = TrajectoryLoss(output=output_val_xyz, trajectory_gt=output_trajectory_val_xyz[..., :-1], mask=output_trajectory_val_mask[..., :-1], lengths=output_trajectory_val_lengths) + GravityLoss(output=output_val_xyz.clone(), trajectory_gt=output_trajectory_val_xyz[..., :-1], mask=output_trajectory_val_mask[..., :-1], lengths=output_trajectory_val_lengths) + EndOfTrajectoryLoss(output_eot=output_val_eot, eot_gt=input_trajectory_val_gt[..., -1], mask=input_trajectory_val_mask[..., -1], lengths=input_trajectory_val_lengths, eot_startpos=input_trajectory_val_startpos[..., -1], flag='Validation')


  print('Generator Loss : {:.3f}'.format(loss_G.item()), end=', ')
  print('Discriminator Loss : {:.3f}'.format(loss_D.item()), end=', ')
  print('Val Loss : {:.3f}'.format(val_loss.item()))
  print('======> [Generator] Adversarial Loss ({}) : {:.3f}'.format(args.adversarial_lossfn, adversarial_loss_G.item()), end=', ')
  print('Trajectory Loss : {:.3f}'.format(trajectory_loss.item()), end=', ')
  print('Gravity Loss : {:.3f}'.format(gravity_loss.item()))
  print('EndOfTrajectory Loss : {:.3f}'.format(eot_loss.item()))
  print('======> [Discriminator] Adversarial Loss ({}) : {:.3f}'.format(args.adversarial_lossfn, loss_D.item()))
  wandb.log({'Generator Loss':loss_G.item(), 'Discriminator Loss':loss_D.item(), 'Validation Loss':val_loss.item()})

  if visualize_trajectory_flag == True and vis_signal == True:
    make_visualize(input_trajectory_train=input_trajectory_train, output_train_depth=output_train_depth, input_trajectory_val=input_trajectory_val, output_val_depth=output_val_depth, output_train_xyz=output_train_xyz, output_trajectory_train_xyz=output_trajectory_train_xyz, output_trajectory_train_startpos=output_trajectory_train_startpos, input_trajectory_train_lengths=input_trajectory_train_lengths, output_trajectory_train_maks=output_trajectory_train_mask, output_val_xyz=output_val_xyz, output_trajectory_val_xyz=output_trajectory_val_xyz, output_trajectory_val_startpos=output_trajectory_val_startpos, input_trajectory_val_lengths=input_trajectory_val_lengths, output_trajectory_val_mask=output_trajectory_val_mask, visualization_path=visualization_path)

  return loss_G.item(), loss_D.item(), val_loss.item(), generator_eot, generator_depth, discriminator

def initialize_folder(path):
  if not os.path.exists(path):
      os.makedirs(path)

def collate_fn_padd(batch):
    '''
    Padding batch of variable length
    '''
    padding_value = -10
    ## Get sequence lengths
    lengths = pt.tensor([trajectory[1:, :].shape[0] for trajectory in batch])
    # Input features : columns 4-5 contain u, v in screen space
    ## Padding 
    input_batch = [pt.Tensor(trajectory[1:, [4, 5, -2]]) for trajectory in batch] # (4, 5, -2) = (u, v ,end_of_trajectory)
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=padding_value)
    ## Retrieve initial position (u, v, depth)
    input_startpos = pt.stack([pt.Tensor(trajectory[0, [4, 5, 6, -2]]) for trajectory in batch])  # (4, 5, 6, -2) = (u, v, depth, end_of_trajectory)
    input_startpos = pt.unsqueeze(input_startpos, dim=1)
    ## Compute mask
    input_mask = (input_batch != padding_value)

    # Output features : columns 6 cotain depth from camera to projected screen
    ## Padding
    output_batch = [pt.Tensor(trajectory[:, [6, -2]]) for trajectory in batch]
    output_batch = pad_sequence(output_batch, batch_first=True)
    ## Retrieve initial position
    output_startpos = pt.stack([pt.Tensor(trajectory[0, [0, 1, 2, -2]]) for trajectory in batch])
    output_startpos = pt.unsqueeze(output_startpos, dim=1)
    ## Retrieve the x, y, z in world space for compute the reprojection error (x', y', z' <===> x, y, z)
    output_xyz = [pt.Tensor(trajectory[:, [0, 1, 2, -2]]) for trajectory in batch]
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
  if model_arch=='bigru_residual_adversarial':
    generator_eot = Generator(input_size=2, output_size=1)
    generator_depth = Generator(input_size=3, output_size=1)
    discriminator = Discriminator(input_size=3, output_size=1)
  else :
    print("Please input correct model architecture : gru, bigru, lstm, bilstm")
    exit()

  return generator_eot, generator_depth, discriminator

def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D):
  if args.load_checkpoint == 'best':
    load_checkpoint = '{}/{}/{}_best.pth'.format(args.save_checkpoint + args.wandb_tags.replace('/', '_'), args.wandb_notes, args.wandb_notes)
  elif args.load_checkpoint == 'lastest':
    load_checkpoint = '{}/{}/{}_lastest.pth'.format(args.save_checkpoint + args.wandb_tags.replace('/', '_'), args.wandb_notes, args.wandb_notes)
  else:
    print("[#] The load_checkpoint should be \'best\' or \'lastest\' keywords...")
    exit()

  if os.path.isfile(load_checkpoint):
    print("[#] Found the checkpoint...")
    checkpoint = pt.load(load_checkpoint, map_location='cuda:0')
    # Load optimizer, learning rate, decay and scheduler parameters
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
    lr_scheduler_D.load_state_dict(checkpoint['lr_scheduler_D'])
    start_epoch = checkpoint['epoch']
    min_val_loss = checkpoint['min_val_loss']
    return generator, discriminator, optimizer_G, optimizer_D, start_epoch, lr_scheduler_G, lr_scheduler_D, min_val_loss

  else:
    print("[#] Checkpoint not found...")
    exit()

if __name__ == '__main__':
  print('[#]Training : Trajectory Estimation')
  # Argumentparser for input
  parser = argparse.ArgumentParser(description='Predict the 3D projectile')
  parser.add_argument('--dataset_train_path', dest='dataset_train_path', type=str, help='Path to training set', required=True)
  parser.add_argument('--dataset_val_path', dest='dataset_val_path', type=str, help='Path to validation set', required=True)
  parser.add_argument('--batch_size', dest='batch_size', type=int, help='Size of batch', default=50)
  parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, help='Type of trajectory(Rolling, Projectile, MagnusProjectile)', default='Projectile')
  parser.add_argument('--no_visualize', dest='visualize_trajectory_flag', help='No Visualize the trajectory', action='store_false')
  parser.add_argument('--visualize', dest='visualize_trajectory_flag', help='Visualize the trajectory', action='store_true')
  parser.add_argument('--save_checkpoint', dest='save_checkpoint', type=str, help='Path to save a model checkpoint', required=True)
  parser.add_argument('--load_checkpoint', dest='load_checkpoint', type=str, help='Path to load a trained model checkpoint', default=None)
  parser.add_argument('--visualization_path', dest='visualization_path', type=str, help='Path to visualization directory', default='./visualize_html/')
  parser.add_argument('--cam_params_file', dest='cam_params_file', type=str, help='Path to camera parameters file(Intrinsic/Extrinsic)')
  parser.add_argument('--wandb_name', dest='wandb_name', type=str, help='WanDB session name', default=None)
  parser.add_argument('--wandb_tags', dest='wandb_tags', type=str, help='WanDB tags name', default=None)
  parser.add_argument('--cuda_device_num', dest='cuda_device_num', type=int, help='Provide cuda device number', default=0)
  parser.add_argument('--wandb_notes', dest='wandb_notes', type=str, help='WanDB notes', default="")
  parser.add_argument('--model_arch', dest='model_arch', type=str, help='Input the model architecture(lstm, bilstm, gru, bigru)', required=True)
  parser.add_argument('--clip', dest='clip', type=float, help='Clipping gradients value', required=True)
  parser.add_argument('--noise', dest='noise', help='Noise on the fly', action='store_true')
  parser.add_argument('--no_noise', dest='noise', help='Noise on the fly', action='store_false')
  parser.add_argument('--noise_sd', dest='noise_sd', help='Std. of noise', type=float, default=None)
  parser.add_argument('--pretrained_generator', help='Pretraining the generator before training the discriminator', action='store_true')
  parser.add_argument('--no_pretrained_generator', help='Pretraining the generator before training the discriminator', action='store_false')
  parser.add_argument('--adversarial_lossfn', help='Specify the loss fucntion of aderverial training', type=str, default='gan')
  parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
  parser.add_argument('--wandb_dir', help='Path to WanDB directory', type=str, default='./')
  args = parser.parse_args()

  # Init wandb
  wandb.init(project="ball-trajectory-estimation", name=args.wandb_name, tags=args.wandb_tags, notes=args.wandb_notes, dir=args.wandb_dir)

  # Initialize folder
  initialize_folder(args.visualization_path)
  save_checkpoint = '{}/{}/'.format(args.save_checkpoint + args.wandb_tags.replace('/', '_'), args.wandb_notes)
  initialize_folder(save_checkpoint)

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

  # Create Datasetloader for train and validation
  print(args.dataset_train_path)
  trajectory_train_dataset = TrajectoryDataset(dataset_path=args.dataset_train_path, trajectory_type=args.trajectory_type)
  trajectory_train_dataloader = DataLoader(trajectory_train_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True, collate_fn=collate_fn_padd, pin_memory=True, drop_last=True)
  # Create Datasetloader for validation
  trajectory_val_dataset = TrajectoryDataset(dataset_path=args.dataset_val_path, trajectory_type=args.trajectory_type)
  trajectory_val_dataloader = DataLoader(trajectory_val_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True, collate_fn=collate_fn_padd, pin_memory=True, drop_last=True)
  # Cast it to iterable object
  trajectory_val_iterloader = iter(trajectory_val_dataloader)

  # Dataset format
  # Trajectory path : (x0, y0) ... (xn, yn)
  print("======================================================Summary Batch (batch_size = {})=========================================================================".format(args.batch_size))
  for key, batch in enumerate(trajectory_train_dataloader):
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
  min_val_loss = 2e10
  generator_eot, generator_depth, discriminator = get_model(input_size=n_input, output_size=n_output, model_arch=args.model_arch)
  generator_eot = generator_eot.to(device)
  generator_depth = generator_depth.to(device)
  discriminator = discriminator.to(device)

  # Define optimizer, learning rate, decay and scheduler parameters
  learning_rate = args.lr
  params_G = list(generator_eot.parameters()) + list(generator_depth.parameters())
  optimizer_G = pt.optim.Adam(params_G, lr=learning_rate)
  optimizer_D = pt.optim.Adam(discriminator.parameters(), lr=learning_rate)
  decay_rate = 0.5
  lr_scheduler_G = pt.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_G, gamma=decay_rate)
  lr_scheduler_D = pt.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_D, gamma=decay_rate)
  start_epoch = 1

  # Load the checkpoint if it's available.
  if args.load_checkpoint is None:
    # Create a model
    print('===>No model checkpoint')
    print('[#] Define the Learning rate, Optimizer, Decay rate and Scheduler...')
  else:
    print('===>Load checkpoint with Optimizer state, Decay and Scheduler state')
    print('[#] Loading ... {}'.format(args.load_checkpoint))
    generator, discriminator, optimizer_G, optimizer_D, start_epoch, lr_scheduler_G, lr_scheduler_D, min_val_loss = load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D)


  print('[#]Model Architecture')
  print('####### Generator - EOT #######')
  print(generator_eot)
  print('####### Generator - Depth #######')
  print(generator_depth)
  print('####### Discriminator #######')
  print(discriminator)

  # Log metrics with wandb
  wandb.watch(generator_eot)
  wandb.watch(generator_depth)
  wandb.watch(discriminator)

  # Training settings
  n_epochs = 100000
  decay_cycle = 400
  counter = 0 # Value can be 1-15, depends on n_epoch training generator and discriminator
  total_epochs_train_generator = 0
  total_epochs_train_discriminator = 0
  n_train_generator = 20
  n_train_discriminator = 10
  pretrained_generator = args.pretrained_generator
  n_pretrained_generator = 5
  for epoch in range(start_epoch, n_epochs+1):
    accumulate_discriminator_loss = []
    accumulate_generator_loss = []
    accumulate_val_loss = []
    # Fetch the Validation set (Get each batch for each training epochs)
    try:
      batch_val = next(trajectory_val_iterloader)
    except StopIteration:
      trajectory_val_iterloader = iter(trajectory_val_dataloader)
      batch_val = next(trajectory_val_iterloader)
    input_trajectory_val = batch_val['input'][0].to(device)
    input_trajectory_val_lengths = batch_val['input'][1].to(device)
    input_trajectory_val_mask = batch_val['input'][2].to(device)
    input_trajectory_val_startpos = batch_val['input'][3].to(device)
    output_trajectory_val = batch_val['output'][0].to(device)
    output_trajectory_val_lengths = batch_val['output'][1].to(device)
    output_trajectory_val_mask = batch_val['output'][2].to(device)
    output_trajectory_val_startpos = batch_val['output'][3].to(device)
    output_trajectory_val_xyz = batch_val['output'][4].to(device)

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>[Epoch : {}/{}]<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(epoch, n_epochs))

    # Log the learning rate
    for param_group in optimizer_D.param_groups:
      print("[#]Learning rate (Discriminator) : ", param_group['lr'])
      wandb.log({'Learning Rate (Discriminator)':param_group['lr']})
    for param_group in optimizer_G.param_groups:
      print("[#]Learning rate (Generator) : ", param_group['lr'])
      wandb.log({'Learning Rate (Generator)':param_group['lr']})

    # Visualize signal to make a plot and save to wandb every epoch is done.
    # vis_signal = True if batch_idx+1 == len(trajectory_train_dataloader) else False
    vis_signal = True if epoch % 1 == 0 else False

    # This pretrained the generator for first few epochs
    if epoch > n_pretrained_generator and pretrained_generator:
      pretrained_generator = False
      counter = 1

    # Training signal for swapping training between generator and discriminator
    if pretrained_generator:
      train_discriminator = False
      train_generator = True
      print("[#]Pretraining a Generator...[{}/{}]".format(epoch, n_pretrained_generator))
    elif counter>=1 and counter<=n_train_discriminator:
      train_discriminator = True
      train_generator = False
      total_epochs_train_discriminator += 1
      print("[#]Training a Discriminator...[{}/{}]".format(counter, n_train_discriminator))
    elif counter>n_train_discriminator and counter<=n_train_generator+n_train_discriminator:
      train_discriminator = False
      train_generator = True
      total_epochs_train_generator += 1
      print("[#]Training a Generator...[{}/{}]".format(counter-n_train_discriminator, n_train_generator))
    else :
      counter = 1
      train_discriminator = True
      train_generator = False
      total_epochs_train_discriminator += 1
      print("[#]Training Discriminator...[{}/{}]".format(counter, n_train_discriminator))

    # Training a model iterate over dataloader to get each batch and pass to train function
    for batch_idx, batch_train in enumerate(trajectory_train_dataloader):
      print('===> [Minibatch {}/{}].........'.format(batch_idx+1, len(trajectory_train_dataloader)), end='')
      # Training set (Each index in batch_train came from the collate_fn_padd)
      input_trajectory_train = batch_train['input'][0].to(device)
      input_trajectory_train_lengths = batch_train['input'][1].to(device)
      input_trajectory_train_mask = batch_train['input'][2].to(device)
      input_trajectory_train_startpos = batch_train['input'][3].to(device)
      output_trajectory_train = batch_train['output'][0].to(device)
      output_trajectory_train_lengths = batch_train['output'][1].to(device)
      output_trajectory_train_mask = batch_train['output'][2].to(device)
      output_trajectory_train_startpos = batch_train['output'][3].to(device)
      output_trajectory_train_xyz = batch_train['output'][4].to(device)


      # Call function to train
      loss_G, loss_D, val_loss, generator_eot, generator_depth, discriminator = train(output_trajectory_train=output_trajectory_train, output_trajectory_train_mask=output_trajectory_train_mask,
                                                                 output_trajectory_train_lengths=output_trajectory_train_lengths, output_trajectory_train_startpos=output_trajectory_train_startpos, output_trajectory_train_xyz=output_trajectory_train_xyz,
                                                                 input_trajectory_train=input_trajectory_train, input_trajectory_train_mask = input_trajectory_train_mask,
                                                                 input_trajectory_train_lengths=input_trajectory_train_lengths, input_trajectory_train_startpos=input_trajectory_train_startpos,
                                                                 output_trajectory_val=output_trajectory_val, output_trajectory_val_mask=output_trajectory_val_mask,
                                                                 output_trajectory_val_lengths=output_trajectory_val_lengths, output_trajectory_val_startpos=output_trajectory_val_startpos,
                                                                 input_trajectory_val=input_trajectory_val, input_trajectory_val_mask=input_trajectory_val_mask, output_trajectory_val_xyz=output_trajectory_val_xyz,
                                                                 input_trajectory_val_lengths=input_trajectory_val_lengths, input_trajectory_val_startpos=input_trajectory_val_startpos,
                                                                 generator_eot=generator_eot, generator_depth=generator_depth, discriminator=discriminator,
                                                                 visualize_trajectory_flag=args.visualize_trajectory_flag, train_generator=train_generator, train_discriminator=train_discriminator,
                                                                 projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix,
                                                                 optimizer_G=optimizer_G, optimizer_D=optimizer_D, epoch=epoch, n_epochs=n_epochs, vis_signal=vis_signal, width=width, height=height)

      accumulate_val_loss.append(val_loss)
      accumulate_generator_loss.append(loss_G)
      accumulate_discriminator_loss.append(loss_D)
      vis_signal = False

    counter += 1
    # Get the average loss for each epoch over entire dataset
    val_loss_per_epoch = np.mean(accumulate_val_loss)
    generator_loss_per_epoch = np.mean(accumulate_generator_loss)
    discriminator_loss_per_epoch = np.mean(accumulate_discriminator_loss)
    # Log the each epoch loss
    wandb.log({'Epoch Generator Loss':generator_loss_per_epoch, 'Epoch Discriminator Loss':discriminator_loss_per_epoch, 'Epoch Validation Loss':val_loss_per_epoch, 'Total #N training generator' : total_epochs_train_generator, 'Total #N training discriminator' : total_epochs_train_discriminator})

    # Decrease learning rate every n_epochs % decay_cycle batch
    if total_epochs_train_generator % decay_cycle == 0 and train_generator and not pretrained_generator:
      lr_scheduler_G.step()
      for param_group in optimizer_G.param_groups:
        print("Generator - Stepping Learning rate to ", param_group['lr'])
    if total_epochs_train_discriminator % decay_cycle == 0 and train_discriminator:
      lr_scheduler_D.step()
      for param_group in optimizer_D.param_groups:
        print("Discriminator - Stepping Learning rate to ", param_group['lr'])

    # Save the model checkpoint every finished the epochs
    print('[#]Finish Epoch : {}/{}.........Generator loss : {:.3f}, Discriminator loss : {:.3f}, Val loss : {:.3f}'.format(epoch, n_epochs, generator_loss_per_epoch, discriminator_loss_per_epoch, val_loss_per_epoch))
    if min_val_loss > val_loss_per_epoch:
      # Save model checkpoint
      save_checkpoint_best = '{}/{}_best.pth'.format(save_checkpoint, args.wandb_notes)
      print('[+++]Saving the best model checkpoint : Prev loss {:.3f} > Curr loss {:.3f}'.format(min_val_loss, val_loss_per_epoch))
      print('[+++]Saving the best model checkpoint to : ', save_checkpoint_best)
      min_val_loss = val_loss_per_epoch
      # Save to directory
      checkpoint = {'epoch':epoch+1, 'generator':generator.state_dict(), 'discriminator':discriminator.state_dict(), 'optimizer_G':optimizer_G.state_dict(), 'optimizer_D':optimizer_D.state_dict(), 'lr_scheduler_D':lr_scheduler_D.state_dict(), 'lr_scheduler_G':lr_scheduler_G.state_dict(), 'min_val_loss':min_val_loss}
      pt.save(checkpoint, save_checkpoint_best)
      pt.save(checkpoint, os.path.join(wandb.run.dir, 'checkpoint_best.pth'))

    else:
      print('[#]Not saving the best model checkpoint : Val loss {:.3f} not improved from {:.3f}'.format(val_loss_per_epoch, min_val_loss))


    if epoch % 20 == 0:
      # Save the lastest checkpoint for continue training every 10 epoch
      save_checkpoint_lastest = '{}/{}_lastest.pth'.format(save_checkpoint, args.wandb_notes)
      print('[#]Saving the lastest checkpoint to : ', save_checkpoint_lastest)
      checkpoint = {'epoch':epoch+1, 'generator':generator.state_dict(), 'discriminator':discriminator.state_dict(), 'optimizer_G':optimizer_G.state_dict(), 'optimizer_D':optimizer_D.state_dict(), 'lr_scheduler_D':lr_scheduler_D.state_dict(), 'lr_scheduler_G':lr_scheduler_G.state_dict(), 'min_val_loss':min_val_loss}
      pt.save(checkpoint, save_checkpoint_lastest)
      pt.save(checkpoint, os.path.join(wandb.run.dir, 'checkpoint_lastest.pth'))

  print("[#] Done")
