from __future__ import print_function
# Import libs
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
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
import utils.transformation as utils_transform
# Loss
import loss

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
parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('--decay_gamma', help='Gamma (Decay rate)', type=float, default=0.8)
parser.add_argument('--decay_cycle', help='Decay cycle', type=int, default=150)
parser.add_argument('--teacherforcing_depth', help='Use a teacher forcing training scheme for depth displacement estimation', action='store_true', default=False)
parser.add_argument('--teacherforcing_mixed', help='Use a teacher forcing training scheme for depth displacement estimation on some part of training set', action='store_true', default=False)
parser.add_argument('--wandb_dir', help='Path to WanDB directory', type=str, default='./')
parser.add_argument('--start_decumulate', help='Epoch to start training with decumulate of an error', type=int, default=0)
parser.add_argument('--decumulate', help='Decumulate the depth by ray casting', action='store_true', default=False)
args = parser.parse_args()

# GPU initialization
if pt.cuda.is_available():
  pt.cuda.set_device(args.cuda_device_num)
  device = pt.device('cuda')
  print('[%]GPU Enabled')
else:
  device = pt.device('cpu')
  print('[%]GPU Disabled, CPU Enabled')

# Init wandb
wandb.init(project="ball-trajectory-estimation", name=args.wandb_name, tags=args.wandb_tags, notes=args.wandb_notes, dir=args.wandb_dir)

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
  uv_cumsum = pt.stack([pt.cat([trajectory_startpos[i][:, [0, 1]], uv[i].clone().detach()]) for i in range(trajectory_startpos.shape[0])])
  # trajectory_cumsum : perform cumsum along the sequence_length axis
  uv_cumsum = pt.cumsum(uv_cumsum, dim=1)
  # output : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output = pt.stack([pt.cat([trajectory_startpos[i][:, [2]], depth[i]]) for i in range(trajectory_startpos.shape[0])])
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
  exit()
  '''
  masking_noise = pt.nn.init.uniform_(pt.empty(input_trajectory[..., :-1].shape)).to(device) > np.random.rand(1)[0]
  n_noise = int(args.batch_size * factor)
  noise_idx = np.random.choice(a=args.batch_size, size=(n_noise,), replace=False)
  input_trajectory[noise_idx] += noise_uv[noise_idx] * masking_noise[noise_idx]
  input_trajectory = pt.tensor(np.diff(input_trajectory.cpu().numpy(), axis=1)).to(device)
  return input_trajectory

def get_plane_normal():
  a = pt.tensor([32., 0., 19.])
  b = pt.tensor([32., 0., -31.])
  c = pt.tensor([-28., 0., 19.])
  plane_normal = pt.cross(b-a, c-a)
  return plane_normal.to(device)

def raycasting(reset_idx, uv, lengths, depth, cam_params_dict, plane_normal):
  screen_width = cam_params_dict['width']
  screen_height = cam_params_dict['height']
  I_inv = cam_params_dict['I_inv']
  E_inv = cam_params_dict['E_inv']
  # print(reset_idx, uv, lengths, depth)
  camera_center = E_inv[:-1, -1]
  # Ray casting
  transformation = pt.inverse(pt.inverse(I_inv) @ pt.inverse(E_inv))   # Inverse(Intrinsic @ Extrinsic)
  uv = pt.cat((uv[reset_idx[0], :], pt.ones(uv[reset_idx[0], :].shape).to(device)), dim=-1)
  uv[:, 0] = ((uv[:, 0]/screen_width) * 2) - 1
  uv[:, 1] = ((uv[:, 1]/screen_height) * 2) - 1
  ndc = (uv @ transformation.t()).to(device)
  ray_direction = ndc[:, :-1] - camera_center
  # Depth that intersect the pitch
  plane_point = pt.tensor([32, 0, 19]).to(device)
  distance = camera_center - plane_point
  normalize = pt.tensor([-(pt.dot(distance, plane_normal)/pt.dot(ray_direction[i], plane_normal)) for i in range(ray_direction.shape[0])]).view(-1, 1).to(device)
  intersect_pos = pt.cat(((camera_center - ray_direction * normalize), pt.ones(ray_direction.shape[0], 1).to(device)), dim=-1)
  reset_depth = intersect_pos @ pt.inverse(E_inv).t()
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
  depth_chunk = [depth[start:end] if start == 0 else depth[start+1:end] for start, end in zip(reset_idx, reset_idx[1:])]
  depth_chunk = [pt.cat((reset_depth[i].view(-1, 1), depth_chunk[i])) for i in range(len(depth_chunk))]
  depth_chunk_cumsum = [pt.cumsum(each_depth_chunk, dim=0) for each_depth_chunk in depth_chunk]
  depth_chunk = pt.cat(depth_chunk_cumsum)
  return depth_chunk

def cumsum_decumulate_trajectory(depth, uv, trajectory_startpos, lengths, eot, cam_params_dict):
  # print(depth.shape, uv.shape, trajectory_startpos.shape, eot.shape)
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
  # uv_cumsum : concat with startpos and stack back to (batch_size, sequence_length+1, 2)
  uv_cumsum = pt.stack([pt.cat([trajectory_startpos[i][:, :2], uv[i].clone().detach()]) for i in range(trajectory_startpos.shape[0])])
  # uv_cumsum : perform cumsum along the sequence_length axis
  uv_cumsum = pt.cumsum(uv_cumsum, dim=1)
  # Reset the depth when eot == 1
  plane_normal = get_plane_normal()

  eot_all = pt.stack([pt.cat([trajectory_startpos[i][:, -1].view(-1, 1), eot[i]]) for i in range(trajectory_startpos.shape[0])])
  reset_idx = [pt.where((eot_all[i][:lengths[i]+1]) == 1.) for i in range(eot_all.shape[0])]
  reset_depth = [raycasting(reset_idx=reset_idx[i], depth=depth[i], uv=uv_cumsum[i], lengths=lengths[i], cam_params_dict=cam_params_dict, plane_normal=plane_normal) for i in range(trajectory_startpos.shape[0])]
  # output : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  depth_cumsum = [split_cumsum(reset_idx=reset_idx[i][0]+1, length=lengths[i], reset_depth=reset_depth[i], start_pos=trajectory_startpos[i], depth=depth[i], eot=eot_all[i]) for i in range(trajectory_startpos.shape[0])]
  depth_cumsum = pt.stack(depth_cumsum, dim=0)
  return depth_cumsum, uv_cumsum

def cumsum_trajectory_teacherforcing(depth, depth_teacher, uv, trajectory_startpos):
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
  # Teacher forcing use the ground truth depth
  depth_teacher = pt.stack([pt.cat([trajectory_startpos[i][:, -1], depth_teacher[i][:, 0]]) for i in range(trajectory_startpos.shape[0])])
  depth_teacher = pt.cumsum(depth_teacher, dim=1).unsqueeze(dim=-1)
  # Apply cummulative summation to output
  # trajectory_cumsum : concat with startpos and stack back to (batch_size, sequence_length+1, 2)
  uv_cumsum = pt.stack([pt.cat([trajectory_startpos[i][:, :2], uv[i].clone().detach()]) for i in range(trajectory_startpos.shape[0])])
  # trajectory_cumsum : perform cumsum along the sequence_length axis
  uv_cumsum = pt.cumsum(uv_cumsum, dim=1)
  # output : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  depth_cumsum = depth + depth_teacher[:, :-1, :]
  depth_cumsum = pt.stack([pt.cat([trajectory_startpos[i][:, -1], depth_cumsum[i][:, 0]]) for i in range(trajectory_startpos.shape[0])])
  # output = pt.stack([pt.cat([trajectory_startpos[i][:, -1].view(-1, 1), depth[i]]) for i in range(trajectory_startpos.shape[0])])
  # output : perform cumsum along the sequence_length axis
  return pt.unsqueeze(depth_cumsum, dim=-1), uv_cumsum

def train(output_trajectory_train, output_trajectory_train_mask, output_trajectory_train_lengths, output_trajectory_train_startpos, output_trajectory_train_xyz, input_trajectory_train, input_trajectory_train_mask, input_trajectory_train_lengths, input_trajectory_train_startpos, model_flag, model_depth, output_trajectory_val, output_trajectory_val_mask, output_trajectory_val_lengths, output_trajectory_val_startpos, output_trajectory_val_xyz, input_trajectory_val, input_trajectory_val_mask, input_trajectory_val_lengths, input_trajectory_val_startpos, epoch, vis_signal, optimizer, cam_params_dict, visualize_trajectory_flag=True, visualization_path='./visualize_html/'):
  # Training RNN/LSTM model
  # Run over each example
  # Train a model
  # Initial the state for EOT and Depth
  hidden_eot = model_flag.initHidden(batch_size=args.batch_size)
  cell_state_eot = model_flag.initCellState(batch_size=args.batch_size)
  hidden_depth = model_depth.initHidden(batch_size=args.batch_size)
  cell_state_depth = model_depth.initCellState(batch_size=args.batch_size)

  # Training mode
  model_flag.train()
  model_depth.train()
  # Add noise on the fly
  input_trajectory_train_gt = input_trajectory_train.clone()
  input_trajectory_val_gt = input_trajectory_val.clone()
  if args.noise:
    input_trajectory_train = add_noise(input_trajectory=input_trajectory_train[..., [0, 1]], startpos=input_trajectory_train_startpos[..., [0, 1]], lengths=input_trajectory_train_lengths)
    # input_trajectory_train_eot = input_trajectory_train[..., [0, 1]].clone()
    input_trajectory_val = add_noise(input_trajectory=input_trajectory_val[..., [0, 1]], startpos=input_trajectory_val_startpos[..., [0, 1]], lengths=input_trajectory_val_lengths)
    # input_trajectory_val_eot = input_trajectory_val[..., [0, 1]].clone()

  # Forward PASSING
  # Forward pass for training a model
  # Predict the EOT
  output_train_eot, (_, _) = model_flag(input_trajectory_train, hidden_eot, cell_state_eot, lengths=input_trajectory_train_lengths)
  output_train_eot = pt.sigmoid(output_train_eot).clone()
  input_trajectory_train = pt.cat((input_trajectory_train, output_train_eot, input_trajectory_train_gt[..., [3]]), dim=2)
  # input_trajectory_train = pt.cat((input_trajectory_train[..., :-1], (output_train_eot > 0.5).type(pt.cuda.FloatTensor)), dim=2)
  # Predict the DEPTH
  output_train_depth, (_, _) = model_depth(input_trajectory_train, hidden_depth, cell_state_depth, lengths=input_trajectory_train_lengths)

  # De-accumulate module
  # (This step we get the displacement of depth by input the displacement of u and v)
  # Apply cummulative summation to output using cumsum_trajectory function
  if args.decumulate and epoch > args.start_decumulate:
    output_train_depth_cumsum, input_trajectory_train_uv_cumsum = cumsum_decumulate_trajectory(depth=output_train_depth, uv=input_trajectory_train_gt[..., [0, 1]], trajectory_startpos=input_trajectory_train_startpos, lengths=input_trajectory_train_lengths, eot=input_trajectory_train_gt[..., [2]], cam_params_dict=cam_params_dict)
    # output_train_depth, input_trajectory_train_uv = cumsum_decumulate_trajectory(depth=output_train_depth, uv=input_trajectory_train_gt[..., :-1], trajectory_startpos=input_trajectory_train_startpos, lengths=input_trajectory_train_lengths, eot=(output_train_eot > 0.5).type(pt.cuda.FloatTensor), projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, width=width, height=height)

  elif args.teacherforcing_depth:
    output_train_depth_cumsum, input_trajectory_train_uv_cumsum = cumsum_trajectory_teacherforcing(depth=output_train_depth, depth_teacher=output_trajectory_train, uv=input_trajectory_train_gt[..., [0, 1]], trajectory_startpos=input_trajectory_train_startpos[..., [0, 1, 2]])

  elif args.teacherforcing_mixed:
    factor = np.random.uniform(low=0.3, high=0.6)
    n_teacherforcing = int(args.batch_size * factor)
    print(input_trajectory_train_gt.shape, input_trajectory_train_startpos.shape)
    teacher_idx = np.random.choice(a=args.batch_size, size=(n_teacherforcing,), replace=False)
    print(teacher_idx)
    output_train_depth_cumsum, input_trajectory_train_uv_cumsum = cumsum_trajectory(depth=output_train_depth, uv=input_trajectory_train_gt[..., [0, 1]], trajectory_startpos=input_trajectory_train_startpos[..., [0, 1, 2]])
    print(input_trajectory_train_gt.shape, input_trajectory_train_startpos.shape)
    print(input_trajectory_train_gt[[0, 1], :, [0, 1]].shape)
    print(input_trajectory_train_startpos[teacher_idx, :, [0, 1]])
    print(input_trajectory_train_gt[teacher_idx, :, [0, 1]])
    exit()
    output_train_depth_cumsum[teacher_idx, ...], _ = cumsum_trajectory_teacherforcing(depth=output_train_depth[teacher_idx, ...], depth_teacher=output_trajectory_train[teacher_idx, ...], uv=input_trajectory_train_gt[teacher_idx, :, [0, 1]], trajectory_startpos=input_trajectory_train_startpos[teacher_idx, :, [0, 1, 2]])

  else:
    output_train_depth_cumsum, input_trajectory_train_uv_cumsum = cumsum_trajectory(depth=output_train_depth, uv=input_trajectory_train_gt[..., [0, 1]], trajectory_startpos=input_trajectory_train_startpos[..., [0, 1, 2]])

  # Project the (u, v, depth) to world space
  output_train_xyz = pt.stack([utils_transform.projectToWorldSpace(uv=input_trajectory_train_uv_cumsum[i], depth=output_train_depth_cumsum[i], cam_params_dict=cam_params_dict, device=device) for i in range(output_train_depth.shape[0])])

  ####################################
  ############# EOT&Depth ############
  ####################################
  optimizer.zero_grad() # Clear existing gradients from previous epoch
  train_trajectory_loss = loss.TrajectoryLoss(output=output_train_xyz, trajectory_gt=output_trajectory_train_xyz[..., [0, 1, 2]], mask=output_trajectory_train_mask[..., [0, 1, 2]], lengths=output_trajectory_train_lengths)
  train_gravity_loss = loss.GravityLoss(output=output_train_xyz, trajectory_gt=output_trajectory_train_xyz[..., [0, 1, 2]], mask=output_trajectory_train_mask[..., [0, 1, 2]], lengths=output_trajectory_train_lengths)
  train_eot_loss = loss.EndOfTrajectoryLoss(output_eot=output_train_eot, eot_gt=input_trajectory_train_gt[..., [2]], mask=input_trajectory_train_mask[..., [2]], lengths=input_trajectory_train_lengths, eot_startpos=input_trajectory_train_startpos[..., [3]], flag='Train')
  train_depth_loss = loss.DepthLoss(output=output_train_depth, depth_gt=output_trajectory_train[..., [0]], lengths=input_trajectory_train_lengths, mask=input_trajectory_train_mask)
  train_below_ground_loss = loss.BelowGroundPenalize(output=output_train_xyz, trajectory_gt=output_trajectory_train_xyz[..., [0, 1, 2]], mask=output_trajectory_train_mask[..., [0, 1, 2]], lengths=output_trajectory_train_lengths)

  # Sum up all train loss 
  train_loss = train_trajectory_loss + train_eot_loss*100 + train_depth_loss*1000 + train_gravity_loss + train_below_ground_loss
  train_loss.backward()
  for name, p in model_flag.named_parameters():
    # print(name, p.grad)
    p.data.clamp_(-args.clip, args.clip)
  for name, p in model_depth.named_parameters():
    # print(name, p.grad)
    p.data.clamp_(-args.clip, args.clip)
  optimizer.step()

  # Evaluating mode
  model_flag.eval()
  model_depth.eval()
  # Forward pass for validate a model
  output_val_eot, (_, _) = model_flag(input_trajectory_val, hidden_eot, cell_state_eot, lengths=input_trajectory_val_lengths)
  output_val_eot = pt.sigmoid(output_val_eot).clone()
  input_trajectory_val = pt.cat((input_trajectory_val, output_val_eot, input_trajectory_val_gt[..., [3]]), dim=2)
  # Predict the DEPTH
  output_val_depth, (_, _) = model_depth(input_trajectory_val, hidden_depth, cell_state_depth, lengths=input_trajectory_val_lengths)

  # De-accumulate module
  if args.decumulate and epoch > args.start_decumulate:
    output_val_depth_cumsum, input_trajectory_val_uv_cumsum = cumsum_decumulate_trajectory(depth=output_val_depth, uv=input_trajectory_val_gt[..., [0, 1]], trajectory_startpos=input_trajectory_val_startpos, lengths=input_trajectory_val_lengths, eot=input_trajectory_val_gt[..., [2]], cam_params_dict=cam_params_dict)
    # output_val_depth, input_trajectory_val_uv = cumsum_decumulate_trajectory(depth=output_val_depth, uv=input_trajectory_val_gt[..., :-1], trajectory_startpos=input_trajectory_val_startpos, lengths=input_trajectory_val_lengths, eot=(output_val_eot > 0.5).type(pt.cuda.FloatTensor), projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, width=width, height=height)
  # elif args.teacherforcing_depth:
    # output_val_depth_cumsum, input_trajectory_val_uv_cumsum = cumsum_trajectory_teacherforcing(depth=output_val_depth, depth_teacher=output_trajectory_val, uv=input_trajectory_val_gt[..., :-1], trajectory_startpos=input_trajectory_val_startpos[..., :-1])
  else:
    # (This step we get the displacement of depth by input the displacement of u and v)
    # Apply cummulative summation to output using cumsum_trajectory function
    output_val_depth_cumsum, input_trajectory_val_uv_cumsum = cumsum_trajectory(depth=output_val_depth, uv=input_trajectory_val_gt[..., [0, 1]], trajectory_startpos=input_trajectory_val_startpos[..., [0, 1, 2]])

  # Project the (u, v, depth) to world space
  output_val_xyz = pt.stack([utils_transform.projectToWorldSpace(uv=input_trajectory_val_uv_cumsum[i], depth=output_val_depth_cumsum[i], cam_params_dict=cam_params_dict, device=device) for i in range(output_val_depth.shape[0])])

  # Calculate loss of unprojected trajectory
  val_trajectory_loss = loss.TrajectoryLoss(output=output_val_xyz, trajectory_gt=output_trajectory_val_xyz[..., [0, 1, 2]], mask=output_trajectory_val_mask[..., [0, 1, 2]], lengths=output_trajectory_val_lengths)
  val_gravity_loss = loss.GravityLoss(output=output_val_xyz, trajectory_gt=output_trajectory_val_xyz[..., [0, 1, 2]], mask=output_trajectory_val_mask[..., [0, 1, 2]], lengths=output_trajectory_val_lengths)
  val_eot_loss = loss.EndOfTrajectoryLoss(output_eot=output_val_eot, eot_gt=input_trajectory_val_gt[..., [2]], mask=input_trajectory_val_mask[..., [2]], lengths=input_trajectory_val_lengths, eot_startpos=input_trajectory_val_startpos[..., [3]], flag='Validation')
  val_depth_loss = loss.DepthLoss(output=output_val_depth, depth_gt=output_trajectory_val[..., [0]], lengths=input_trajectory_val_lengths, mask=input_trajectory_val_mask)
  val_below_ground_loss = loss.BelowGroundPenalize(output=output_val_xyz, trajectory_gt=output_trajectory_val_xyz[..., [0, 1, 2]], mask=output_trajectory_val_mask[..., [0, 1, 2]], lengths=output_trajectory_val_lengths)
  val_loss = val_trajectory_loss + val_gravity_loss + val_eot_loss*100 + val_below_ground_loss + val_depth_loss*1000

  print('Train Loss : {:.3f}'.format(train_loss.item()), end=', ')
  print('Val Loss : {:.3f}'.format(val_loss.item()))
  print('======> Trajectory Loss : {:.3f}'.format(train_trajectory_loss.item()), end=', ')
  print('Gravity Loss : {:.3f}'.format(train_gravity_loss.item()), end=', ')
  print('EndOfTrajectory Loss : {:.3f}'.format(train_eot_loss.item()), end=', ')
  print('BelowGroundPenalize Loss : {:.3f}'.format(train_below_ground_loss.item()), end=', ')
  print('Depth Loss : {:.3f}'.format(train_depth_loss.item()))
  wandb.log({'Train Loss':train_loss.item(), 'Validation Loss':val_loss.item()})

  if visualize_trajectory_flag == True and vis_signal == True:
    utils_func.make_visualize(input_trajectory_train=input_trajectory_train, output_train_depth=output_train_depth, input_trajectory_val=input_trajectory_val, output_val_depth=output_val_depth, output_train_xyz=output_train_xyz, output_trajectory_train_xyz=output_trajectory_train_xyz, output_trajectory_train_startpos=output_trajectory_train_startpos, input_trajectory_train_lengths=input_trajectory_train_lengths, output_trajectory_train_mask=output_trajectory_train_mask, output_val_xyz=output_val_xyz, output_trajectory_val_xyz=output_trajectory_val_xyz, output_trajectory_train_lengths=output_trajectory_train_lengths, output_trajectory_val_lengths=output_trajectory_val_lengths, output_trajectory_val_startpos=output_trajectory_val_startpos, input_trajectory_val_lengths=input_trajectory_val_lengths, output_trajectory_val_mask=output_trajectory_val_mask, visualization_path=visualization_path)

  return train_loss.item(), val_loss.item(), model_flag, model_depth

def collate_fn_padd(batch):
    '''
    Padding batch of variable length
    Columns convention : (x, y, z, u, v, d, eot, og, rad)
    '''
    x, y, z, u, v, d, eot, og, rad, g = range(10)
    padding_value = -10
    ## Get sequence lengths
    lengths = pt.tensor([trajectory[1:, :].shape[0] for trajectory in batch])
    # Input features : columns 4-5 contain u, v in screen space
    ## Padding 
    input_batch = [pt.Tensor(trajectory[1:, [u, v, eot, rad]]) for trajectory in batch] # (4, 5, -2) = (u, v ,end_of_trajectory)
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=padding_value)
    ## Retrieve initial position (u, v, depth)
    input_startpos = pt.stack([pt.Tensor(trajectory[0, [u, v, d, eot]]) for trajectory in batch])  # (4, 5, 6, -2) = (u, v, depth, end_of_trajectory)
    input_startpos = pt.unsqueeze(input_startpos, dim=1)
    ## Compute mask
    input_mask = (input_batch != padding_value)

    # Output features : columns 6 cotain depth from camera to projected screen
    ## Padding
    output_batch = [pt.Tensor(trajectory[1:, [d, eot]]) for trajectory in batch]
    output_batch = pad_sequence(output_batch, batch_first=True)
    ## Retrieve initial position
    output_startpos = pt.stack([pt.Tensor(trajectory[0, [x, y, z, eot]]) for trajectory in batch])
    output_startpos = pt.unsqueeze(output_startpos, dim=1)
    ## Retrieve the x, y, z in world space for compute the reprojection error (x', y', z' <===> x, y, z)
    output_xyz = [pt.Tensor(trajectory[:, [x, y, z, eot]]) for trajectory in batch]
    output_xyz = pad_sequence(output_xyz, batch_first=True, padding_value=padding_value)
    ## Compute mask
    output_mask = (output_xyz != padding_value)
    ## Compute cummulative summation to form a trajectory from displacement every columns except the end_of_trajectory
    # print(output_xyz[..., :-1].shape, pt.unsqueeze(output_xyz[..., -1], dim=2).shape)
    output_xyz = pt.cat((pt.cumsum(output_xyz[..., :-1], dim=1), pt.unsqueeze(output_xyz[..., -1], dim=2)), dim=2)

    # print("Mask shape : ", mask.shape)
    return {'input':[input_batch, lengths, input_mask, input_startpos],
            'output':[output_batch, lengths+1, output_mask, output_startpos, output_xyz]}

def load_checkpoint(model_flag, model_depth, optimizer, lr_scheduler):
  if args.load_checkpoint == 'best':
    load_checkpoint = '{}/{}/{}_best.pth'.format(args.save_checkpoint + args.wandb_tags.replace('/', '_'), args.wandb_name, args.wandb_name)
  elif args.load_checkpoint == 'lastest':
    load_checkpoint = '{}/{}/{}_lastest.pth'.format(args.save_checkpoint + args.wandb_tags.replace('/', '_'), args.wandb_name, args.wandb_name)
  else:
    print("[#] The load_checkpoint should be \'best\' or \'lastest\' keywords...")
    exit()

  if os.path.isfile(load_checkpoint):
    print("[#] Found the checkpoint...")
    checkpoint = pt.load(load_checkpoint, map_location='cuda:0')
    # Load optimizer, learning rate, decay and scheduler parameters
    model_flag.load_state_dict(checkpoint['model_flag'])
    model_depth.load_state_dict(checkpoint['model_depth'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch']
    min_val_loss = checkpoint['min_val_loss']
    return model_flag, model_depth, optimizer, start_epoch, lr_scheduler, min_val_loss

  else:
    print("[#] Checkpoint not found...")
    exit()

if __name__ == '__main__':
  print('[#]Training : Trajectory Estimation')

  # Initialize folder
  utils_func.initialize_folder(args.visualization_path)
  save_checkpoint = '{}/{}/'.format(args.save_checkpoint + args.wandb_tags.replace('/', '_'), args.wandb_name)
  utils_func.initialize_folder(save_checkpoint)

  # Load camera parameters : ProjectionMatrix and WorldToCameraMatrix
  cam_params_dict = utils_transform.get_cam_params_dict(args.cam_params_file, device)

  # Create Datasetloader for train and validation
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
  min_val_loss = 2e10
  model_flag, model_depth = utils_func.get_model(model_arch=args.model_arch)
  model_flag = model_flag.to(device)
  model_depth = model_depth.to(device)

  # Define optimizer, learning rate, decay and scheduler parameters
  params = list(model_flag.parameters()) + list(model_depth.parameters())
  optimizer = pt.optim.Adam(params, lr=args.lr)
  lr_scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.decay_gamma)

  # Load the checkpoint if it's available.
  if args.load_checkpoint is None:
    # Create a model
    print('===>No model checkpoint')
    print('[#] Define the Learning rate, Optimizer, Decay rate and Scheduler...')
  else:
    print('===>Load checkpoint with Optimizer state, Decay and Scheduler state')
    print('[#] Loading ... {}'.format(args.load_checkpoint))
    model_flag, model_depth, optimizer, start_epoch, lr_scheduler, min_val_loss = load_checkpoint(model_flag, model_depth, optimizer, lr_scheduler)

  print('[#]Model Architecture')
  print('####### Model - EOT #######')
  print(model_flag)
  print('####### Model - Depth #######')
  print(model_depth)

  # Log metrics with wandb
  wandb.watch(model_flag)
  wandb.watch(model_depth)

  # Training settings
  n_epochs = 100000
  start_epoch = 1
  for epoch in range(start_epoch, n_epochs+1):
    accumulate_train_loss = []
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
    for param_group in optimizer.param_groups:
      print("[#]Learning rate (Depth & EOT) : ", param_group['lr'])
      wandb.log({'Learning Rate (Depth & EOT)':param_group['lr']})

    # Visualize signal to make a plot and save to wandb every epoch is done.
    vis_signal = True if epoch % 1 == 0 else False

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
      train_loss, val_loss, model_flag, model_depth = train(output_trajectory_train=output_trajectory_train, output_trajectory_train_mask=output_trajectory_train_mask,
                                                                 output_trajectory_train_lengths=output_trajectory_train_lengths, output_trajectory_train_startpos=output_trajectory_train_startpos,
                                                                 output_trajectory_train_xyz=output_trajectory_train_xyz, input_trajectory_train=input_trajectory_train,
                                                                 input_trajectory_train_mask = input_trajectory_train_mask, input_trajectory_train_lengths=input_trajectory_train_lengths,
                                                                 input_trajectory_train_startpos=input_trajectory_train_startpos, output_trajectory_val=output_trajectory_val,
                                                                 output_trajectory_val_mask=output_trajectory_val_mask, output_trajectory_val_lengths=output_trajectory_val_lengths,
                                                                 output_trajectory_val_startpos=output_trajectory_val_startpos, input_trajectory_val=input_trajectory_val,
                                                                 input_trajectory_val_mask=input_trajectory_val_mask, output_trajectory_val_xyz=output_trajectory_val_xyz,
                                                                 input_trajectory_val_lengths=input_trajectory_val_lengths, input_trajectory_val_startpos=input_trajectory_val_startpos,
                                                                 model_flag=model_flag, model_depth=model_depth, visualize_trajectory_flag=args.visualize_trajectory_flag,
                                                                 optimizer=optimizer, epoch=epoch, vis_signal=vis_signal, cam_params_dict=cam_params_dict, visualization_path=args.visualization_path)

      accumulate_val_loss.append(val_loss)
      accumulate_train_loss.append(train_loss)
      vis_signal = False

    # Get the average loss for each epoch over entire dataset
    val_loss_per_epoch = np.mean(accumulate_val_loss)
    train_loss_per_epoch = np.mean(accumulate_train_loss)
    # Log the each epoch loss
    wandb.log({'Epoch Train Loss':train_loss_per_epoch, 'Epoch Validation Loss':val_loss_per_epoch})

    # Decrease learning rate every n_epochs % decay_cycle batch
    if epoch % args.decay_cycle == 0:
      lr_scheduler.step()
      for param_group in optimizer.param_groups:
        print("Stepping Learning rate to ", param_group['lr'])

    # Save the model checkpoint every finished the epochs
    print('[#]Finish Epoch : {}/{}.........Train loss : {:.3f}, Val loss : {:.3f}'.format(epoch, n_epochs, train_loss_per_epoch, val_loss_per_epoch))
    if min_val_loss > val_loss_per_epoch:
      # Save model checkpoint
      save_checkpoint_best = '{}/{}_best.pth'.format(save_checkpoint, args.wandb_name)
      print('[+++]Saving the best model checkpoint : Prev loss {:.3f} > Curr loss {:.3f}'.format(min_val_loss, val_loss_per_epoch))
      print('[+++]Saving the best model checkpoint to : ', save_checkpoint_best)
      min_val_loss = val_loss_per_epoch
      # Save to directory
      checkpoint = {'epoch':epoch+1, 'model_depth':model_depth.state_dict(), 'model_flag':model_flag.state_dict(), 'optimizer':optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict(), 'min_val_loss':min_val_loss}
      pt.save(checkpoint, save_checkpoint_best)
      pt.save(checkpoint, os.path.join(wandb.run.dir, 'checkpoint_best.pth'))

    else:
      print('[#]Not saving the best model checkpoint : Val loss {:.3f} not improved from {:.3f}'.format(val_loss_per_epoch, min_val_loss))


    if epoch % 20 == 0:
      # Save the lastest checkpoint for continue training every 10 epoch
      save_checkpoint_lastest = '{}/{}_lastest.pth'.format(save_checkpoint, args.wandb_name)
      print('[#]Saving the lastest checkpoint to : ', save_checkpoint_lastest)
      checkpoint = {'epoch':epoch+1, 'model_depth':model_depth.state_dict(), 'model_flag':model_flag.state_dict(), 'optimizer':optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict(), 'min_val_loss':min_val_loss}
      pt.save(checkpoint, save_checkpoint_lastest)
      pt.save(checkpoint, os.path.join(wandb.run.dir, 'checkpoint_lastest.pth'))

  print("[#] Done")
