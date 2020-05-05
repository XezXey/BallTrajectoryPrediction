from __future__ import print_function
# Import libs
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import glob
import os
import argparse
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import PIL.Image
import io
import wandb
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from mpl_toolkits import mplot3d
import json
# Dataloader
from utils.dataloader import TrajectoryDataset
# Models
from models.rnn_model import RNN
from models.lstm_model import LSTM
from models.bilstm_model import BiLSTM
from models.gru_model import GRU
from models.bigru_model import BiGRU

def visualize_layout_update(fig=None, n_vis=5):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  for i in range(n_vis*2):
    fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-50, 50],), yaxis = dict(nticks=5, range=[-2, 20],), zaxis = dict(nticks=10, range=[-30, 30],),)
  return fig

def make_visualize(output_train_xyz, output_trajectory_train_xyz, output_trajectory_train_startpos, input_trajectory_train_lengths, output_trajectory_train_maks, output_val_xyz, output_trajectory_val_xyz, output_trajectory_val_startpos, input_trajectory_val_lengths, output_trajectory_val_mask, visualization_path):
  # Visualize by make a subplots of trajectory
  n_vis = 5
  # Random the index the be visualize
  train_vis_idx = np.random.randint(low=0, high=input_trajectory_train_startpos.shape[0], size=(n_vis))
  val_vis_idx = np.random.randint(low=0, high=input_trajectory_val_startpos.shape[0], size=(n_vis))
  fig = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  # Append the start position and apply cummulative summation for transfer the displacement to the x, y, z coordinate. These will done by visualize_trajectory function
  # Can use mask directly because the mask obtain from full trajectory(Not remove the start pos)
  visualize_trajectory(output=pt.mul(output_train_xyz, output_trajectory_train_mask[..., :-1]), trajectory_gt=output_trajectory_train_xyz[..., :-1], trajectory_startpos=output_trajectory_train_startpos[..., :-1], lengths=input_trajectory_train_lengths, mask=output_trajectory_train_mask[..., :-1], fig=fig, flag='Train', n_vis=n_vis, vis_idx=train_vis_idx)
  visualize_trajectory(output=pt.mul(output_val_xyz, output_trajectory_val_mask[..., :-1]), trajectory_gt=output_trajectory_val_xyz[..., :-1], trajectory_startpos=output_trajectory_val_startpos[..., :-1], lengths=input_trajectory_val_lengths, mask=output_trajectory_val_mask[..., :-1], fig=fig, flag='Validation', n_vis=n_vis, vis_idx=val_vis_idx)
  # Adjust the layout/axis
  # For an AUTO SCALED
  fig.update_layout(height=1920, width=1500, autosize=True)
  plotly.offline.plot(fig, filename='./{}/trajectory_visualization_depth_auto_scaled.html'.format(visualization_path), auto_open=False)
  wandb.log({"AUTO SCALED : Trajectory Visualization(Col1=Train, Col2=Val)":wandb.Html(open('./{}/trajectory_visualization_depth_auto_scaled.html'.format(visualization_path)))})
  # For a PITCH SCALED
  fig = visualize_layout_update(fig=fig, n_vis=n_vis)
  plotly.offline.plot(fig, filename='./{}/trajectory_visualization_depth_pitch_scaled.html'.format(visualization_path), auto_open=False)
  wandb.log({"PITCH SCALED : Trajectory Visualization(Col1=Train, Col2=Val)":wandb.Html(open('./{}/trajectory_visualization_depth_pitch_scaled.html'.format(visualization_path)))})

def visualize_trajectory(output, trajectory_gt, trajectory_startpos, lengths, mask, vis_idx, fig=None, flag='train', n_vis=5):
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
    fig.add_trace(go.Scatter3d(x=output[i][:lengths[i]+1, 0], y=output[i][:lengths[i]+1, 1], z=output[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}".format(flag, i, MSELoss(pt.tensor(output[i]).to(device), pt.tensor(trajectory_gt[i]).to(device), mask=mask[i]))), row=idx+1, col=col)
    fig.add_trace(go.Scatter3d(x=trajectory_gt[i][:lengths[i]+1, 0], y=trajectory_gt[i][:lengths[i]+1, 1], z=trajectory_gt[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+1, col=col)

def compute_gravity_constraint_penalize(output, trajectory_gt, mask, lengths):
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
    gravity_constraint_penalize += ((pt.sum(trajectory_gt_yaxis_2nd_finite_difference - output_yaxis_2nd_finite_difference)))**2
  return gravity_constraint_penalize

def compute_below_ground_constraint_penalize(output, mask, lengths):
  # Penalize when the y-axis is below on the ground
  output = output * mask
  below_ground_constraint_penalize = pt.sum((output[:, :, 1][output[:, :, 1] < -1])**2)
  return below_ground_constraint_penalize

def MSELoss(output, trajectory_gt, mask, lengths=None, delmask=True):
  if lengths is None :
    gravity_constraint_penalize = pt.tensor(0).to(device)
    # below_ground_constraint_penalize = pt.tensor(0).to(device)
  else:
    # Penalize the model if predicted values are not fall by gravity(2nd derivatives)
    gravity_constraint_penalize = compute_gravity_constraint_penalize(output=output.clone(), trajectory_gt=trajectory_gt.clone(), mask=mask, lengths=lengths)
    # Penalize the model if predicted values are below the ground (y < 0)
    # below_ground_constraint_penalize = compute_below_ground_constraint_penalize(output=output.clone(), mask=mask, lengths=lengths)
  mse_loss = (pt.sum((((trajectory_gt - output))**2) * mask) / pt.sum(mask)) + gravity_constraint_penalize # + below_ground_constraint_penalize
  return mse_loss

def projectToWorldSpace(screen_space, depth, projection_matrix, camera_to_world_matrix):
  # print(screen_space.shape, depth.shape)
  depth = depth.view(-1)
  screen_width = 1920.
  screen_height = 1080.
  screen_space = pt.div(screen_space, pt.tensor([screen_width, screen_height]).to(device)) # Normalize : (width, height) -> (-1, 1)
  screen_space = (screen_space * 2.0) - pt.ones(size=(screen_space.size()), dtype=pt.float32).to(device) # Normalize : (width, height) -> (-1, 1)
  screen_space = (screen_space.t() * depth).t()   # Normalize : (-1, 1) -> (-depth, depth)
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


def train(output_trajectory_train, output_trajectory_train_mask, output_trajectory_train_lengths, output_trajectory_train_startpos, output_trajectory_train_xyz, input_trajectory_train, input_trajectory_train_mask, input_trajectory_train_lengths, input_trajectory_train_startpos, model, output_trajectory_val, output_trajectory_val_mask, output_trajectory_val_lengths, output_trajectory_val_startpos, output_trajectory_val_xyz, input_trajectory_val, input_trajectory_val_mask, input_trajectory_val_lengths, input_trajectory_val_startpos, hidden, cell_state, projection_matrix, camera_to_world_matrix, epoch, n_epochs, vis_signal, optimizer, visualize_trajectory_flag=True, visualization_path='./visualize_html/'):
  # Training RNN/LSTM model
  # Run over each example
  # Initial hidden layer for the first RNN Cell
  # Train a model
  hidden = rnn_model.initHidden(batch_size=args.batch_size)
  cell_state = rnn_model.initCellState(batch_size=args.batch_size)
  # Training mode
  model.train()
  optimizer.zero_grad() # Clear existing gradients from previous epoch
  # Forward PASSING
  # Forward pass for training a model  
  output_train, (_, _) = model(input_trajectory_train, hidden, cell_state, lengths=input_trajectory_train_lengths)
  # (This step we get the displacement of depth by input the displacement of u and v)
  # Apply cummulative summation to output using cumsum_trajectory function
  output_train, input_trajectory_train_temp = cumsum_trajectory(output=output_train, trajectory=input_trajectory_train[..., :-1], trajectory_startpos=input_trajectory_train_startpos[..., :-1])
  # Project the (u, v, depth) to world space
  output_train_xyz = pt.stack([projectToWorldSpace(screen_space=input_trajectory_train_temp[i], depth=output_train[i], projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix) for i in range(output_train.shape[0])])
  # Evaluating mode
  model.eval()
  # Forward pass for validate a model
  output_val, (_, _) = model(input_trajectory_val, hidden, cell_state, lengths=input_trajectory_val_lengths)
  # (This step we get the displacement of depth by input the displacement of u and v)
  # Apply cummulative summation to output using cumsum_trajectory function
  output_val, input_trajectory_val_temp = cumsum_trajectory(output=output_val, trajectory=input_trajectory_val[..., :-1], trajectory_startpos=input_trajectory_val_startpos[..., :-1])
  # Project the (u, v, depth) to world space
  output_val_xyz = pt.stack([projectToWorldSpace(screen_space=input_trajectory_val_temp[i], depth=output_val[i], projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix) for i in range(output_val.shape[0])])
  # Detach for use hidden as a weights in next batch
  cell_state.detach()
  cell_state = cell_state.detach()
  hidden.detach()
  hidden = hidden.detach()

  # Calculate loss of unprojected trajectory
  train_loss = MSELoss(output=output_train_xyz, trajectory_gt=output_trajectory_train_xyz[..., :-1], mask=output_trajectory_train_mask[..., :-1], lengths=output_trajectory_train_lengths)
  val_loss = MSELoss(output=output_val_xyz, trajectory_gt=output_trajectory_val_xyz[..., :-1], mask=output_trajectory_val_mask[..., :-1], lengths=output_trajectory_val_lengths)

  train_loss.backward() # Perform a backpropagation and calculates gradients
  pt.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
  optimizer.step() # Updates the weights accordingly to the gradients

  print('Train Loss : {:.3f}'.format(train_loss.item()), end=', ')
  print('Val Loss : {:.3f}'.format(val_loss.item()))
  wandb.log({'Train Loss':train_loss.item(), 'Validation Loss':val_loss.item()})

  if visualize_trajectory_flag == True and vis_signal == True:
    make_visualize(output_train_xyz=output_train_xyz, output_trajectory_train_xyz=output_trajectory_train_xyz, output_trajectory_train_startpos=output_trajectory_train_startpos, input_trajectory_train_lengths=input_trajectory_train_lengths, output_trajectory_train_maks=output_trajectory_train_mask, output_val_xyz=output_val_xyz, output_trajectory_val_xyz=output_trajectory_val_xyz, output_trajectory_val_startpos=output_trajectory_val_startpos, input_trajectory_val_lengths=input_trajectory_val_lengths, output_trajectory_val_mask=output_trajectory_val_mask, visualization_path=visualization_path)

  return train_loss.item(), val_loss.item(), hidden, cell_state, model

def initialize_folder(path):
  if not os.path.exists(path):
      os.makedirs(path)

def collate_fn_padd(batch):
    '''
    Padding batch of variable length
    '''
    ## Get sequence lengths
    lengths = pt.tensor([trajectory[1:, :].shape[0] for trajectory in batch])
    # Input features : columns 4-5 contain u, v in screen space
    ## Padding 
    input_batch = [pt.Tensor(trajectory[1:, [4, 5, -2]]) for trajectory in batch] # (4, 5, -1) = (u, v ,end_of_trajectory)
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=-1)
    ## Retrieve initial position (u, v, depth)
    input_startpos = pt.stack([pt.Tensor(trajectory[0, [4, 5, 6, -2]]) for trajectory in batch])  # (4, 5, 6, -2) = (u, v, depth, end_of_trajectory)
    input_startpos = pt.unsqueeze(input_startpos, dim=1)
    ## Compute mask
    input_mask = (input_batch != -1)

    # Output features : columns 6 cotain depth from camera to projected screen
    ## Padding
    output_batch = [pt.Tensor(trajectory[:, [6, -2]]) for trajectory in batch]
    output_batch = pad_sequence(output_batch, batch_first=True)
    ## Retrieve initial position
    output_startpos = pt.stack([pt.Tensor(trajectory[0, [0, 1, 2, -2]]) for trajectory in batch])
    output_startpos = pt.unsqueeze(output_startpos, dim=1)
    ## Retrieve the x, y, z in world space for compute the reprojection error (x', y', z' <===> x, y, z)
    output_xyz = [pt.Tensor(trajectory[:, [0, 1, 2, -2]]) for trajectory in batch]
    output_xyz = pad_sequence(output_xyz, batch_first=True, padding_value=-1)
    ## Compute mask
    output_mask = (output_xyz != -1)
    ## Compute cummulative summation to form a trajectory from displacement every columns except the end_of_trajectory
    # print(output_xyz[..., :-1].shape, pt.unsqueeze(output_xyz[..., -1], dim=2).shape)
    output_xyz = pt.cat((pt.cumsum(output_xyz[..., :-1], dim=1), pt.unsqueeze(output_xyz[..., -1], dim=2)), dim=2)

    # print("Mask shape : ", mask.shape)
    return {'input':[input_batch, lengths, input_mask, input_startpos],
            'output':[output_batch, lengths, output_mask, output_startpos, output_xyz]}

def get_model(input_size, output_size, model_arch):
  if model_arch=='gru':
    rnn_model = GRU(input_size=input_size, output_size=output_size)
  elif model_arch=='bigru':
    rnn_model = BiGRU(input_size=input_size, output_size=output_size)
  elif model_arch=='lstm':
    rnn_model = LSTM(input_size=input_size, output_size=output_size)
  elif model_arch=='bilstm':
    rnn_model = BiLSTM(input_size=input_size, output_size=output_size)
  else :
    print("Please input correct model architecture : gru, bigru, lstm, bilstm")
    exit()

  return rnn_model

if __name__ == '__main__':
  print('[#]Training : Trajectory Estimation')
  # Argumentparser for input
  parser = argparse.ArgumentParser(description='Predict the 2D projectile')
  parser.add_argument('--dataset_train_path', dest='dataset_train_path', type=str, help='Path to training set', required=True)
  parser.add_argument('--dataset_val_path', dest='dataset_val_path', type=str, help='Path to validation set', required=True)
  parser.add_argument('--batch_size', dest='batch_size', type=int, help='Size of batch', default=50)
  parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, help='Type of trajectory(Rolling, Projectile, MagnusProjectile)', default='Projectile')
  parser.add_argument('--no_visualize', dest='visualize_trajectory_flag', help='No Visualize the trajectory', action='store_false')
  parser.add_argument('--visualize', dest='visualize_trajectory_flag', help='Visualize the trajectory', action='store_true')
  parser.add_argument('--model_checkpoint_path', dest='model_checkpoint_path', type=str, help='Path to save a model checkpoint', required=True)
  parser.add_argument('--model_path', dest='model_path', type=str, help='Path to load a trained model checkpoint', default=None)
  parser.add_argument('--visualization_path', dest='visualization_path', type=str, help='Path to visualization directory', default='./visualize_html/')
  parser.add_argument('--cam_params_file', dest='cam_params_file', type=str, help='Path to camera parameters file(Intrinsic/Extrinsic)')
  parser.add_argument('--wandb_name', dest='wandb_name', type=str, help='WanDB session name', default=None)
  parser.add_argument('--wandb_tags', dest='wandb_tags', type=str, help='WanDB tags name', default=None)
  parser.add_argument('--cuda_device_num', dest='cuda_device_num', type=int, help='Provide cuda device number', default=0)
  parser.add_argument('--wandb_notes', dest='wandb_notes', type=str, help='WanDB notes', default="")
  parser.add_argument('--model_arch', dest='model_arch', type=str, help='Input the model architecture(lstm, bilstm, gru, bigru)', required=True)
  args = parser.parse_args()

  # Init wandb
  wandb.init(project="ball-trajectory-estimation", name=args.wandb_name, tags=args.wandb_tags, notes=args.wandb_notes)

  # Initialize folder
  initialize_folder(args.visualization_path)
  model_checkpoint_path = '{}/'.format(args.model_checkpoint_path + args.wandb_tags.replace('/', '_'))
  print(model_checkpoint_path)
  initialize_folder(model_checkpoint_path)
  model_checkpoint_path = '{}/{}.pth'.format(model_checkpoint_path, args.wandb_notes)
  print(model_checkpoint_path)

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
    cam_params = dict({'projectionMatrix':cam_params_file['mainCameraParams']['projectionMatrix'], 'worldToCameraMatrix':cam_params_file['mainCameraParams']['worldToCameraMatrix']})
  projection_matrix = np.array(cam_params['projectionMatrix']).reshape(4, 4)
  projection_matrix = pt.tensor([projection_matrix[0, :], projection_matrix[1, :], projection_matrix[3, :], [0, 0, 0, 1]], dtype=pt.float32)
  projection_matrix = pt.inverse(projection_matrix).to(device)
  camera_to_world_matrix = pt.inverse(pt.tensor(cam_params['worldToCameraMatrix']).view(4, 4)).to(device)

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
    unpacked = pt.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, padding_value=-1)
    print("Unpacked equality : ", pt.eq(batch['input'][0], unpacked[0]).all())
    print("===============================================================================================================================================================")

  # Model definition
  n_output = 1 # Contain the depth information of the trajectory
  n_input = 3 # Contain following this trajectory parameters (u, v, end_of_trajectory) position from tracking
  min_val_loss = 2e10
  print('[#]Model Architecture')
  rnn_model = get_model(input_size=n_input, output_size=n_output, model_arch=args.model_arch)
  if args.model_path is None:
    # Create a model
    print('===>No trained model')
  else:
    print('===>Load trained model')
    rnn_model.load_state_dict(pt.load(args.model_path))
  rnn_model = rnn_model.to(device)
  print(rnn_model)

  # Define optimizer parameters
  learning_rate = 0.01
  optimizer = pt.optim.Adam(rnn_model.parameters(), lr=learning_rate)
  decay_rate = 0.96
  lr_scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)
  # Log metrics with wandb
  wandb.watch(rnn_model)

  # Initialize the hidden and cell_state
  hidden = rnn_model.initHidden(batch_size=args.batch_size)
  cell_state = rnn_model.initCellState(batch_size=args.batch_size)

  # Training settings
  n_epochs = 500
  decay_cycle = int(n_epochs/10)
  for epoch in range(1, n_epochs+1):
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
      print("[#]Learning rate : ", param_group['lr'])
      wandb.log({'Learning Rate':param_group['lr']})

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

      # Visualize signal to make a plot and save to wandb
      vis_signal = True if batch_idx+1 == len(trajectory_train_dataloader) else False

      # Call function to train
      train_loss, val_loss, hidden, cell_state, rnn_model = train(output_trajectory_train=output_trajectory_train, output_trajectory_train_mask=output_trajectory_train_mask,
                                                                 output_trajectory_train_lengths=output_trajectory_train_lengths, output_trajectory_train_startpos=output_trajectory_train_startpos, output_trajectory_train_xyz=output_trajectory_train_xyz,
                                                                 input_trajectory_train=input_trajectory_train, input_trajectory_train_mask = input_trajectory_train_mask,
                                                                 input_trajectory_train_lengths=input_trajectory_train_lengths, input_trajectory_train_startpos=input_trajectory_train_startpos,
                                                                 output_trajectory_val=output_trajectory_val, output_trajectory_val_mask=output_trajectory_val_mask,
                                                                 output_trajectory_val_lengths=output_trajectory_val_lengths, output_trajectory_val_startpos=output_trajectory_val_startpos,
                                                                 input_trajectory_val=input_trajectory_val, input_trajectory_val_mask=input_trajectory_val_mask, output_trajectory_val_xyz=output_trajectory_val_xyz,
                                                                 input_trajectory_val_lengths=input_trajectory_val_lengths, input_trajectory_val_startpos=input_trajectory_val_startpos,
                                                                 model=rnn_model, hidden=hidden, cell_state=cell_state, visualize_trajectory_flag=args.visualize_trajectory_flag,
                                                                 projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix,
                                                                 optimizer=optimizer, epoch=epoch, n_epochs=n_epochs, vis_signal=vis_signal)

      accumulate_val_loss.append(val_loss)
      accumulate_train_loss.append(train_loss)

    # Get the average loss for each epoch over entire dataset
    val_loss_per_epoch = np.mean(accumulate_val_loss)
    train_loss_per_epoch = np.mean(accumulate_train_loss)
    # Decrease learning rate every n_epochs % decay_cycle batch
    if n_epochs % decay_cycle == 0:
      lr_scheduler.step()
      for param_group in optimizer.param_groups:
        print("Stepping Learning rate to {}", param_group['lr'])

    # Save the model checkpoint every finished the epochs
    print('[#]Finish Epoch : {}/{}.........Train loss : {:.3f}, Val loss : {:.3f}'.format(epoch, n_epochs, train_loss_per_epoch, val_loss_per_epoch))
    if min_val_loss > val_loss_per_epoch:
      # Save model checkpoint
      print('[#]Saving a model checkpoint : Prev loss {:.3f} > Curr loss {:.3f}'.format(min_val_loss, val_loss_per_epoch))
      min_val_loss = val_loss_per_epoch
      # Save to directory
      pt.save(rnn_model.state_dict(), model_checkpoint_path)
      pt.save(rnn_model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    else:
      print('[#]Not saving a model checkpoint : Val loss {:.3f} not improved from {:.3f}'.format(val_loss_per_epoch, min_val_loss))

  print("[#] Done")
