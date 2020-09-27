from __future__ import print_function
# Import libs
import numpy as np
np.set_printoptions(precision=4)
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

def GravityLoss(output, trajectory_gt, mask, lengths):
  # Compute the 2nd finite difference of the y-axis to get the gravity should be equal in every time step
  gravity_constraint_penalize = pt.tensor([0.])
  count = 0
  # Gaussian blur kernel for not get rid of the input information
  gaussian_blur = pt.tensor([0.25, 0.5, 0.25], dtype=pt.float32).view(1, 1, -1).to(device)
  # Kernel weight for performing a finite difference
  kernel_weight = pt.tensor([-1., 0., 1.], dtype=pt.float32).view(1, 1, -1).to(device)
  # Apply Gaussian blur and finite difference to trajectory_gt
  for i in range(trajectory_gt.shape[0]):
    # print(trajectory_gt[i][:lengths[i]+1, 1])
    # print(trajectory_gt[i][:lengths[i]+1, 1].shape)
    if trajectory_gt[i][:lengths[i], 1].shape[0] < 6:
      print("The trajectory is too shorter to perform a convolution")
      continue
    trajectory_gt_yaxis_1st_gaussian_blur = pt.nn.functional.conv1d(trajectory_gt[i][:lengths[i], 1].view(1, 1, -1), gaussian_blur)
    trajectory_gt_yaxis_1st_finite_difference = pt.nn.functional.conv1d(trajectory_gt_yaxis_1st_gaussian_blur, kernel_weight)
    trajectory_gt_yaxis_2nd_gaussian_blur = pt.nn.functional.conv1d(trajectory_gt_yaxis_1st_finite_difference, gaussian_blur)
    trajectory_gt_yaxis_2nd_finite_difference = pt.nn.functional.conv1d(trajectory_gt_yaxis_2nd_gaussian_blur, kernel_weight)
    # Apply Gaussian blur and finite difference to trajectory_gt
    output_yaxis_1st_gaussian_blur = pt.nn.functional.conv1d(output[i][:lengths[i], 1].view(1, 1, -1), gaussian_blur)
    output_yaxis_1st_finite_difference = pt.nn.functional.conv1d(output_yaxis_1st_gaussian_blur, kernel_weight)
    output_yaxis_2nd_gaussian_blur = pt.nn.functional.conv1d(output_yaxis_1st_finite_difference, gaussian_blur)
    output_yaxis_2nd_finite_difference = pt.nn.functional.conv1d(output_yaxis_2nd_gaussian_blur, kernel_weight)
    # Compute the penalize term
    if gravity_constraint_penalize.shape[0] == 1:
      gravity_constraint_penalize = ((trajectory_gt_yaxis_2nd_finite_difference - output_yaxis_2nd_finite_difference)**2).reshape(-1, 1)
    else:
      gravity_constraint_penalize = pt.cat((gravity_constraint_penalize, ((trajectory_gt_yaxis_2nd_finite_difference - output_yaxis_2nd_finite_difference)**2).reshape(-1, 1)))

  return pt.mean(gravity_constraint_penalize)

def TrajectoryLoss(output, trajectory_gt, mask, lengths=None, delmask=True):
  # L2 loss of reconstructed trajectory
  x_trajectory_loss = (pt.sum((((trajectory_gt[..., 0] - output[..., 0]))**2) * mask[..., 0]) / pt.sum(mask[..., 0]))
  y_trajectory_loss = (pt.sum((((trajectory_gt[..., 1] - output[..., 1]))**2) * mask[..., 1]) / pt.sum(mask[..., 1]))
  z_trajectory_loss = (pt.sum((((trajectory_gt[..., 2] - output[..., 2]))**2) * mask[..., 2]) / pt.sum(mask[..., 2]))
  return x_trajectory_loss + y_trajectory_loss + z_trajectory_loss

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

def cumsum_trajectory(depth, uv, trajectory_startpos):
  '''
  Perform a cummulative summation to the output
  Argument :
  1. depth : The displacement from the network with shape = (batch_size, sequence_length,)
  2. uv : The input_trajectory (displacement of u, v) with shape = (batch_size, sequence_length, 2)
  3. trajectory_startpos : The start position of input trajectory with shape = (batch_size, 1, )
  Output :
  1. output : concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1,)
  2. uv_cumsum : u, v by concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1, 2)
  '''
  # Apply cummulative summation to output
  # trajectory_cumsum : concat with startpos and stack back to (batch_size, sequence_length+1, 2)
  uv_cumsum = pt.stack([pt.cat([trajectory_startpos[i][:, :2], uv[i].clone().detach()]) for i in range(trajectory_startpos.shape[0])])
  # trajectory_cumsum : perform cumsum along the sequence_length axis
  uv_cumsum = pt.cumsum(uv_cumsum, dim=1)
  # output : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output = pt.stack([pt.cat([trajectory_startpos[i][:, -1].view(-1, 1), depth[i]]) for i in range(trajectory_startpos.shape[0])])
  # output : perform cumsum along the sequence_length axis
  depth = pt.cumsum(output, dim=1)
  return depth, uv_cumsum

def add_noise(input_trajectory, startpos, lengths):
  factor = np.random.uniform(low=0.6, high=0.95)
  if args.noise_sd is None:
    noise_sd = np.random.uniform(low=0.3, high=0.7)
  else:
    noise_sd = args.noise_sd
  input_trajectory = pt.cat((startpos, input_trajectory), dim=1)
  input_trajectory = pt.cumsum(input_trajectory, dim=1)
  noise_uv = pt.normal(mean=0.0, std=noise_sd, size=input_trajectory.shape).to(device)
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
  input_trajectory[noise_idx] += noise_uv[noise_idx] * masking_noise[noise_idx]
  input_trajectory = pt.tensor(np.diff(input_trajectory.cpu().numpy(), axis=1)).to(device)
  return input_trajectory

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

def get_plane_normal():
  a = pt.tensor([32., 0., 19.])
  b = pt.tensor([32., 0., -31.])
  c = pt.tensor([-28., 0., 19.])
  plane_normal = pt.cross(b-a, c-a)
  return plane_normal.to(device)

def raycasting(reset_idx, uv, lengths, depth, projection_matrix, camera_to_world_matrix, width, height, plane_normal):
  # print(reset_idx, uv, lengths, depth)
  camera_center = camera_to_world_matrix[:-1, -1]
  # Ray casting
  transformation = pt.inverse(pt.inverse(projection_matrix) @ pt.inverse(camera_to_world_matrix))   # Inverse(Intrinsic @ Extrinsic)
  uv = pt.cat((uv[reset_idx[0], :], pt.ones(uv[reset_idx[0], :].shape).to(device)), dim=-1)
  uv[:, 0] = ((uv[:, 0]/width) * 2) - 1
  uv[:, 1] = ((uv[:, 1]/height) * 2) - 1
  ndc = (uv @ transformation.t()).to(device)
  ray_direction = ndc[:, :-1] - camera_center
  # Depth that intersect the pitch
  plane_point = pt.tensor([32, 0, 19]).to(device)
  distance = camera_center - plane_point
  normalize = pt.tensor([-(pt.dot(distance, plane_normal)/pt.dot(ray_direction[i], plane_normal)) for i in range(ray_direction.shape[0])]).view(-1, 1).to(device)
  intersect_pos = pt.cat(((camera_center - ray_direction * normalize), pt.ones(ray_direction.shape[0], 1).to(device)), dim=-1)
  reset_depth = intersect_pos @ pt.inverse(camera_to_world_matrix).t()
  return reset_depth[..., 2].view(-1, 1)

def split_cumsum(reset_idx, length, start_pos, reset_depth, depth, eot):
  '''
  1. This will split the depth displacement from reset_idx into a chunk. (Ignore the prediction where the EOT=1 in prediction variable. Because we will cast the ray to get that reset depth instead of cumsum to get it.)
  2. Perform cumsum seperately of each chunk.
  3. Concatenate all u, v, depth together and replace with the current one. (Need to replace with padding for masking later on.)
  '''
  reset_idx -= 1
  reset_depth = pt.cat((start_pos[0][2].view(-1, 1), reset_depth))
  max_len = pt.tensor(depth.shape[0]).view(-1, 1).to(device)
  reset_idx = pt.cat((pt.zeros(1).type(pt.cuda.LongTensor).view(-1, 1).to(device), reset_idx.view(-1, 1)))
  if reset_idx[-1] != depth.shape[0] and reset_idx.shape[0] > 1:
    reset_idx = pt.cat((reset_idx, max_len))
  if len(reset_idx) < 2:
    reset_idx = [0, max_len]
  depth_chunk = [depth[start:end] if start == 0 else depth[start+1:end] for start, end in zip(reset_idx, reset_idx[1:])]
  depth_chunk = [pt.cat((reset_depth[i].view(-1, 1), depth_chunk[i])) for i in range(len(depth_chunk))]
  depth_chunk_cumsum = [pt.cumsum(each_depth_chunk, dim=0) for each_depth_chunk in depth_chunk]
  depth_chunk = pt.cat(depth_chunk_cumsum)
  return depth_chunk

def cumsum_decumulate_trajectory(depth, uv, trajectory_startpos, lengths, eot, projection_matrix, camera_to_world_matrix, width, height):
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
  # eot = (eot > 0.5).type(pt.cuda.FloatTensor)
  # Apply cummulative summation to output
  # uv_cumsum : concat with startpos and stack back to (batch_size, sequence_length+1, 2)
  uv_cumsum = pt.stack([pt.cat([trajectory_startpos[i][:, [0, 1]], uv[i].clone().detach()]) for i in range(trajectory_startpos.shape[0])])
  # uv_cumsum : perform cumsum along the sequence_length axis
  uv_cumsum = pt.cumsum(uv_cumsum, dim=1)
  # Reset the depth when eot == 1
  plane_normal = get_plane_normal()

  eot_all = pt.stack([pt.cat([pt.zeros(1, 1).to(device), eot[i]]) for i in range(trajectory_startpos.shape[0])])
  reset_idx = [pt.where((eot_all[i][:lengths[i]+1]) == 1.) for i in range(eot_all.shape[0])]
  reset_depth = [raycasting(reset_idx=reset_idx[i], depth=depth[i], uv=uv_cumsum[i], lengths=lengths[i], projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, width=width, height=height, plane_normal=plane_normal) for i in range(trajectory_startpos.shape[0])]
  # output : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  depth_cumsum = [split_cumsum(reset_idx=reset_idx[i][0]+1, length=lengths[i], reset_depth=reset_depth[i], start_pos=trajectory_startpos[i], depth=depth[i], eot=eot_all[i]) for i in range(trajectory_startpos.shape[0])]
  depth_cumsum = pt.stack(depth_cumsum, dim=0)
  return depth_cumsum, uv_cumsum

def predict_for_all_radian(model, in_f, startpos, h, c, radian, lengths):
  in_f[..., 3] += radian
  output_test_depth, (_, _) = model_depth(in_f, h, c, lengths=lengths)
  if args.decumulate:
    # output_test_depth_cumsum, input_trajectory_test_uv_cumsum = cumsum_decumulate_trajectory(depth=output_test_depth, uv=input_trajectory_test_gt[..., :-1], trajectory_startpos=input_trajectory_test_startpos, lengths=input_trajectory_test_lengths, eot=input_trajectory_test_gt[..., -1].unsqueeze(dim=-1), projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, width=width, height=height)
    output_test_depth_cumsum, input_trajectory_test_uv_cumsum = cumsum_decumulate_trajectory(depth=output_test_depth, uv=in_f[..., [0, 1]], trajectory_startpos=startpos[..., [0, 1, 2]], lengths=lengths, eot=(in_f[..., [2]] > 0.5).type(pt.cuda.FloatTensor), projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, width=width, height=height)

  else:
    output_test_depth_cumsum, input_trajectory_test_uv_cumsum = cumsum_trajectory(depth=output_test_depth, uv=in_f[..., [0, 1]], trajectory_startpos=startpos[..., [0, 1, 2]])
  # Project the (u, v, depth) to world space

  output_test_xyz = pt.stack([projectToWorldSpace(screen_space=input_trajectory_test_uv_cumsum[i], depth=output_test_depth_cumsum[i], projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, width=width, height=height) for i in range(output_test_depth.shape[0])])

  return output_test_xyz

def interactive_optimize(model, in_f, startpos, h, c, lengths, gt, mask):
  # Prediction with given degree
  degree = np.linspace(start=0, stop=360, num=360)
  radian = np.radians(degree)
  output_xyz_all = [predict_for_all_radian(model, in_f, startpos, h, c, each_radian, lengths) for each_radian in radian]
  gt = gt.cpu().detach().numpy()
  for traj_idx in range(in_f.shape[0]):
    trajectory_loss = [TrajectoryLoss(output=output_xyz_all[idx][traj_idx], trajectory_gt=pt.tensor(gt[traj_idx]).to(device), mask=mask[traj_idx], lengths=lengths) for idx in range(degree.shape[0])]
    output_xyz = [output_xyz_all[idx][traj_idx][:lengths[traj_idx], :].cpu().detach().numpy() for idx in range(degree.shape[0])]
    # Build all traces with visible=False
    data_pred = [go.Scatter3d(
      visible = False,
      marker=dict(color='rgba(255, 0, 0, 0.7)', size=5),
      x = output_xyz[idx][:, 0].reshape(-1),
      y = output_xyz[idx][:, 1].reshape(-1),
      z = output_xyz[idx][:, 2].reshape(-1),)
      for idx in range(degree.shape[0])]

    data_pred.append(go.Scatter3d(
      visible = True,
      marker=dict(color='rgba(0, 0, 255, 0.7)', size=5),
      x = gt[traj_idx][:lengths[traj_idx], 0].reshape(-1),
      y = gt[traj_idx][:lengths[traj_idx], 1].reshape(-1),
      z = gt[traj_idx][:lengths[traj_idx], 2].reshape(-1),))


    # Make initial trace visible
    start_index = 0
    data_pred[start_index]['visible'] = True


    # Build slider steps
    steps = []
    for idx, each_degree in enumerate(degree):
      step = dict(
          # Update method allows us to update both trace and layout properties
          method = 'update',
          args = [
              # Make the ith trace visible
              {'visible': [t == each_degree for t in degree]},
              # Set the title for the ith trace
              {'title.text': 'Degree {} with TrajectoryLoss {}'.format(each_degree, trajectory_loss[idx])}],
      )
      step['args'][0]['visible'].append(True) # The ground truth always visible
      steps.append(step)

    # Build sliders
    sliders = [go.layout.Slider(
        active = 10,
        currentvalue = {"prefix": "Degree: "},
        pad = {"t": 50},
        steps = steps
    )]

    layout = go.Layout(
      sliders=sliders,
      title={'text': 'Step {}'.format(start_index)},
      scene=go.layout.Scene(xaxis=dict(nticks=10, range=[-27, 33],), yaxis = dict(nticks=5, range=[-2, 12],), zaxis = dict(nticks=10, range=[-31, 19],), aspectmode='manual', aspectratio=dict(x=4, y=2, z=3))
    )

    fig = go.Figure(data=data_pred, layout=layout)
    fig.show()
    input("Continue plotting...")

def predict(input_trajectory_test, input_trajectory_test_mask, input_trajectory_test_lengths, input_trajectory_test_startpos, model_eot, model_depth, output_trajectory_test, output_trajectory_test_mask, output_trajectory_test_lengths, output_trajectory_test_startpos, output_trajectory_test_xyz, projection_matrix, camera_to_world_matrix, width, height, threshold, trajectory_type, animation_visualize_flag, visualize_trajectory_flag=True, visualization_path='./visualize_html/'):
  # testing RNN/LSTM model
  # Run over each example
  # test a model

  # Initial the state for model and Discriminator for EOT and Depth
  hidden_G_eot = model_eot.initHidden(batch_size=args.batch_size)
  cell_state_G_eot = model_eot.initCellState(batch_size=args.batch_size)
  hidden_G_depth = model_depth.initHidden(batch_size=args.batch_size)
  cell_state_G_depth = model_depth.initCellState(batch_size=args.batch_size)

  # Training mode
  model_eot.eval()
  model_depth.eval()
  # Add noise on the fly
  input_trajectory_test_gt = input_trajectory_test.clone()
  if args.noise:
    input_trajectory_test = add_noise(input_trajectory=input_trajectory_test[..., [0, 1]], startpos=input_trajectory_test_startpos[..., [0, 1]], lengths=input_trajectory_test_lengths)
  else:
    input_trajectory_test = input_trajectory_test[..., [0, 1]]

  # Forward PASSING
  # Forward pass for training a model
  # Predict the EOT
  output_test_eot, (_, _) = model_eot(input_trajectory_test, hidden_G_eot, cell_state_G_eot, lengths=input_trajectory_test_lengths)
  output_test_eot = pt.sigmoid(output_test_eot).clone()
  # Concat du, dv, eot, rad, ...
  input_trajectory_test = pt.cat((input_trajectory_test, output_test_eot, pt.zeros(size=(output_test_eot.shape)).to(device)), dim=2)

  # Interactive Optimize
  interactive_optimize(model=model_depth, in_f=input_trajectory_test, startpos=input_trajectory_test_startpos, h=hidden_G_depth, c=cell_state_G_depth, lengths=input_trajectory_test_lengths, gt=output_trajectory_test_xyz[..., [0, 1, 2]], mask=output_trajectory_test_mask[..., [0, 1, 2]])

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

def get_model(model_arch):
  if model_arch=='bigru_residual_add':
    model_eot = BiGRUResidualAdd(input_size=2, output_size=1)
    model_depth = BiGRUResidualAdd(input_size=4, output_size=1)
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
  parser.add_argument('--save', dest='save', help='Save the prediction trajectory for doing optimization', action='store_true', default=False)
  parser.add_argument('--decumulate', help='Decumulate the depth by ray casting', action='store_true', default=False)

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
  trajectory_test_dataloader = DataLoader(trajectory_test_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True, collate_fn=collate_fn_padd, pin_memory=True, drop_last=False)
  # Create Datasetloader for testidation
  trajectory_test_dataset = TrajectoryDataset(dataset_path=args.dataset_test_path, trajectory_type=args.trajectory_type)
  trajectory_test_dataloader = DataLoader(trajectory_test_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True, collate_fn=collate_fn_padd, pin_memory=True, drop_last=False)
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
  model_eot, model_depth = get_model(model_arch=args.model_arch)
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
    predict(input_trajectory_test=input_trajectory_test, input_trajectory_test_mask = input_trajectory_test_mask,
                                                                 input_trajectory_test_lengths=input_trajectory_test_lengths, input_trajectory_test_startpos=input_trajectory_test_startpos,
                                                                 output_trajectory_test=output_trajectory_test, output_trajectory_test_mask=output_trajectory_test_mask,
                                                                 output_trajectory_test_lengths=output_trajectory_test_lengths, output_trajectory_test_startpos=output_trajectory_test_startpos, output_trajectory_test_xyz=output_trajectory_test_xyz,
                                                                 model_eot=model_eot, model_depth=model_depth,
                                                                 visualize_trajectory_flag=args.visualize_trajectory_flag, projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, threshold=args.threshold, trajectory_type=args.trajectory_type, animation_visualize_flag=args.animation_visualize_flag,
                                                                 width=width, height=height)

  print("[#] Done")