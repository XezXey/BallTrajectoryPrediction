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

def visualize_layout_update(fig=None, n_vis=5):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  for i in range(n_vis*2):
    fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-50, 50],), yaxis = dict(nticks=5, range=[0, 20],), zaxis = dict(nticks=10, range=[-30, 30],),)
  return fig

def visualize_trajectory(output, trajectory_gt, trajectory_startpos, lengths, mask, fig=None, flag='Train', n_vis=5):
  # marker_dict for contain the marker properties
  marker_dict_gt = dict(color='rgba(0, 0, 255, 0.2)', size=3)
  marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=3)
  # detach() for visualization
  output = output.cpu().detach().numpy()
  trajectory_gt = trajectory_gt.cpu().detach().numpy()
  # Random the index the be visualize
  vis_idx = np.random.randint(low=0, high=trajectory_startpos.shape[0], size=(n_vis))
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag=='Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter3d(x=output[i][:lengths[i]+1, 0], y=output[i][:lengths[i]+1, 1], z=output[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}".format(flag, i, MSELoss(pt.tensor(output[i]).to(device), pt.tensor(trajectory_gt[i]).to(device), mask=mask[i]))), row=idx+1, col=col)
    fig.add_trace(go.Scatter3d(x=trajectory_gt[i][:lengths[i]+1, 0], y=trajectory_gt[i][:lengths[i]+1, 1], z=trajectory_gt[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+1, col=col)

def compute_gravity_constraint_penalize(output, trajectory_gt, mask, lengths):
  # Declare a loss
  gravity_constraint_penalize = 0
  # Gaussian blur kernel for not get rid of the input information
  gaussian_blur = pt.tensor([0.25, 0.5, 0.25], dtype=pt.float32).view(1, 1, -1).to(device)
  # Kernel weight for performing a finite difference
  kernel_weight = pt.tensor([-1., 0., 1.], dtype=pt.float32).view(1, 1, -1).to(device)
  # Apply Gaussian blur and finite difference to trajectory_gt and output
  # Use lengths[i]+1 because the lengths calculated from the input space so there's not include the start pos.
  # Once we append the start pos and perform cumsum => the total lengths will equal lengths[i] + 1
  for i in range(trajectory_gt.shape[0]):
    trajectory_gt_yaxis_1st_gaussian_blur = pt.nn.functional.conv1d(trajectory_gt[i][:lengths[i]+1, 1].view(1, 1, -1), gaussian_blur)
    trajectory_gt_yaxis_1st_finite_difference = pt.nn.functional.conv1d(trajectory_gt_yaxis_1st_gaussian_blur, kernel_weight)
    trajectory_gt_yaxis_2nd_gaussian_blur = pt.nn.functional.conv1d(trajectory_gt_yaxis_1st_finite_difference, gaussian_blur)
    trajectory_gt_yaxis_2nd_finite_difference = pt.nn.functional.conv1d(trajectory_gt_yaxis_2nd_gaussian_blur, kernel_weight)
    # Apply Gaussian blur and finite difference to trajectory_gt
    output_yaxis_1st_gaussian_blur = pt.nn.functional.conv1d(output[i][:lengths[i]+1, 1].view(1, 1, -1), gaussian_blur)
    output_yaxis_1st_finite_difference = pt.nn.functional.conv1d(output_yaxis_1st_gaussian_blur, kernel_weight)
    output_yaxis_2nd_gaussian_blur = pt.nn.functional.conv1d(output_yaxis_1st_finite_difference, gaussian_blur)
    output_yaxis_2nd_finite_difference = pt.nn.functional.conv1d(output_yaxis_2nd_gaussian_blur, kernel_weight)
    # print(trajectory_gt_yaxis_2nd_finite_difference, output_yaxis_2nd_finite_difference)
    # exit()
    # Compute the penalize term
    gravity_constraint_penalize += ((pt.sum(trajectory_gt_yaxis_2nd_finite_difference - output_yaxis_2nd_finite_difference)))**2
  return gravity_constraint_penalize

def compute_below_ground_constraint_penalize(output, mask, lengths):
  # Penalize when the y-axis is below on the ground
  output = output * mask
  below_ground_constraint_penalize = pt.sum(pt.abs(output[:, :, 1][output[:, :, 1] < -1]))
  return below_ground_constraint_penalize

def MSELoss(output, trajectory_gt, mask, lengths=None, delmask=True, trajectory_startpos=None):
  if lengths is None:
    gravity_constraint_penalize = pt.tensor(0).to(device)
    # below_ground_constraint_penalize = pt.tensor(0).to(device)
  else:
    # Penalize the model if predicted values are not fall by gravity(2nd derivatives)
    gravity_constraint_penalize = compute_gravity_constraint_penalize(output=output.clone(), trajectory_gt=trajectory_gt.clone(), mask=mask, lengths=lengths)
    # Penalize the model if predicted values are below the ground (y < 0)
    # below_ground_constraint_penalize = compute_below_ground_constraint_penalize(output=output.clone(), mask=mask, lengths=lengths)
  # Calculate MSE Loss
  mse_loss = (pt.sum((((trajectory_gt - output))**2) * mask) / pt.sum(mask)) + gravity_constraint_penalize # + below_ground_constraint_penalize
  return mse_loss

def cumsum_trajectory(trajectory, trajectory_startpos):
  # Stack the start position to the top of each trajectory
  trajectory = pt.stack([pt.cat([trajectory_startpos[i], trajectory[i]]) for i in range(trajectory_startpos.shape[0])])
  # Apply the cummulative summation to get the x, y, z coordinate from the displacement
  trajectory = pt.cumsum(trajectory, dim=1)
  return trajectory

def train(output_trajectory_train, output_trajectory_train_mask, output_trajectory_train_lengths, output_trajectory_train_startpos, output_trajectory_train_xyz, input_trajectory_train, input_trajectory_train_mask, input_trajectory_train_lengths, input_trajectory_train_startpos, model, output_trajectory_val, output_trajectory_val_mask, output_trajectory_val_lengths, output_trajectory_val_startpos, output_trajectory_val_xyz, input_trajectory_val, input_trajectory_val_mask, input_trajectory_val_lengths, input_trajectory_val_startpos, hidden, cell_state, visualize_trajectory_flag=True, min_val_loss=2e10, model_checkpoint_path='./model/', visualization_path='./visualize_html/'):
  # Training RNN/LSTM model 
  n_epochs = 300
  # Train a model
  for epoch in range(1, n_epochs+1):
    hidden = rnn_model.initHidden(batch_size=args.batch_size)
    cell_state = rnn_model.initCellState(batch_size=args.batch_size)
    optimizer.zero_grad() # Clear existing gradients from previous epoch
    # Forward PASSING
    model.train()
    # Forward pass for training a model
    output_train, (_, _) = model(input_trajectory_train, hidden, cell_state, lengths=input_trajectory_train_lengths)
    model.eval()
    # Forward pass for validate a model
    output_val, (_, _) = model(input_trajectory_val, hidden, cell_state, lengths=input_trajectory_val_lengths)
    # Detach for use hidden as a weights in next batch
    cell_state.detach()
    cell_state = cell_state.detach()
    hidden.detach()
    hidden = hidden.detach()
    # Apply cummulative summation to output using cumsum_trajectory function
    output_train = cumsum_trajectory(trajectory=output_train, trajectory_startpos=output_trajectory_train_startpos)
    output_val = cumsum_trajectory(trajectory=output_val, trajectory_startpos=output_trajectory_val_startpos)

    # Calculate loss of reconstructed trajectory
    train_loss = MSELoss(output=output_train, trajectory_gt=output_trajectory_train_xyz, mask=output_trajectory_train_mask, lengths=output_trajectory_train_lengths)
    val_loss = MSELoss(output=output_val, trajectory_gt=output_trajectory_val_xyz, mask=output_trajectory_val_mask, lengths=output_trajectory_val_lengths)

    train_loss.backward() # Perform a backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly to the gradients

    if epoch%10 == 0:
      print('Epoch : {}/{}.........'.format(epoch, n_epochs), end='')
      print('Train Loss : {:.3f}'.format(train_loss.item()), end=', ')
      print('Val Loss : {:.3f}'.format(val_loss.item()))
      wandb.log({'Train Loss':train_loss.item(), 'Validation Loss':val_loss.item()})
      # Save model checkpoint
      if min_val_loss > val_loss:
        print('[#]Saving a model checkpoint')
        min_val_loss = val_loss
        # Save to directory
        pt.save(model.state_dict(), model_checkpoint_path)
        # Save to wandb
        pt.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

    if epoch%100 == 0:
      if visualize_trajectory_flag == True:
        # Visualize by make a subplots of trajectory
        n_vis = 5
        fig = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
        # Append the start position and apply cummulative summation for transfer the displacement to the x, y, z coordinate. These will done by visualize_trajectory function
        # Can use mask directly because the mask obtain from full trajectory(Not remove the start pos)
        visualize_trajectory(output=pt.mul(output_train, output_trajectory_train_mask), trajectory_gt=output_trajectory_train_xyz, trajectory_startpos=output_trajectory_train_startpos, lengths=input_trajectory_train_lengths, mask=output_trajectory_train_mask, fig=fig, flag='Train', n_vis=n_vis)
        visualize_trajectory(output=pt.mul(output_val, output_trajectory_val_mask), trajectory_gt=output_trajectory_val_xyz, trajectory_startpos=output_trajectory_val_startpos, lengths=input_trajectory_val_lengths, mask=output_trajectory_val_mask, fig=fig, flag='Validation', n_vis=n_vis)
        # Adjust the layout/axis
        fig.update_layout(height=1920, width=1500, autosize=True)
        plotly.offline.plot(fig, filename='./{}/trajectory_visualization_xyz_auto_scaled.html'.format(visualization_path), auto_open=False)
        wandb.log({"AUTO SCALED : Trajectory Visualization(Col1=Train, Col2=Val)":wandb.Html(open('./{}/trajectory_visualization_xyz_auto_scaled.html'.format(visualization_path)))})
        fig = visualize_layout_update(fig=fig, n_vis=n_vis)
        plotly.offline.plot(fig, filename='./{}/trajectory_visualization_xyz_pitch_scaled.html'.format(visualization_path), auto_open=False)
        wandb.log({"PITCH SCALED : Trajectory Visualization(Col1=Train, Col2=Val)":wandb.Html(open('./{}/trajectory_visualization_xyz_pitch_scaled.html'.format(visualization_path)))})

  return min_val_loss, hidden, cell_state

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
    input_batch = [pt.Tensor(trajectory[1:, 4:6]) for trajectory in batch]
    input_batch = pt.nn.utils.rnn.pad_sequence(input_batch, batch_first=True, padding_value=-1)
    ## Retrieve initial position (u, v)
    input_startpos = pt.stack([pt.Tensor(trajectory[0, 4:6]) for trajectory in batch])
    input_startpos = pt.unsqueeze(input_startpos, dim=1)
    ## Compute mask
    input_mask = (input_batch != -1)

    # Output features : columns 0-2 cotain x, y, z in world space
    ## Padding
    output_batch = [pt.Tensor(trajectory[1:, :3]) for trajectory in batch]
    output_batch = pt.nn.utils.rnn.pad_sequence(output_batch, batch_first=True)
    ## Retrieve initial position
    output_startpos = pt.stack([pt.Tensor(trajectory[0, :3]) for trajectory in batch])
    output_startpos = pt.unsqueeze(output_startpos, dim=1)
    ## Retrieve the x, y, z in world space for compute the reprojection error (x', y', z' <===> x, y, z)
    output_xyz = [pt.Tensor(trajectory[:, :3]) for trajectory in batch]
    output_xyz = pad_sequence(output_xyz, batch_first=True, padding_value=-1)
    ## Compute mask
    output_mask = (output_xyz != -1)
    output_xyz = pt.cumsum(output_xyz, dim=1)

    # print("Mask shape : ", mask.shape)
    return {'input':[input_batch, lengths, input_mask, input_startpos],
            'output':[output_batch, lengths, output_mask, output_startpos, output_xyz]}

if __name__ == '__main__':
  print('[#]Training : Trajectory Estimation')
  # Argumentparser for input
  parser = argparse.ArgumentParser(description='Predict the 2D projectile')
  parser.add_argument('--dataset_train_path', dest='dataset_train_path', type=str, help='Path to training set', required=True)
  parser.add_argument('--dataset_val_path', dest='dataset_val_path', type=str, help='Path to validation set', required=True)
  parser.add_argument('--batch_size', dest='batch_size', type=int, help='Size of batch', default=50)
  parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, help='Type of trajectory(Rolling, Projectile, MagnusProjectile)', default='Projectile')
  parser.add_argument('--visualize_trajectory_flag', dest='visualize_trajectory_flag', type=bool, help='Visualize the trajectory', default=False)
  parser.add_argument('--model_checkpoint_path', dest='model_checkpoint_path', type=str, help='Path to save a model checkpoint', required=True)
  parser.add_argument('--model_path', dest='model_path', type=str, help='Path to load a trained model checkpoint', default=None)
  parser.add_argument('--visualization_path', dest='visualization_path', type=str, help='Path to visualization directory', default='./visualize_html/')
  parser.add_argument('--wandb_name', dest='wandb_name', type=str, help='WanDB session name', default=None)
  parser.add_argument('--wandb_tags', dest='wandb_tags', type=str, help='WanDB tags name', default=None)
  parser.add_argument('--cuda_device_num', dest='cuda_device_num', type=int, help='Provide cuda device number', default=0)
  parser.add_argument('--wandb_notes', dest='wandb_notes', type=str, help='WanDB notes', default="")
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
  n_output = 3 # Contain the depth information of the trajectory
  n_input = 2 # Contain following this trajectory parameters (u, v) position from tracking
  min_val_loss = 2e10
  print('[#]Model Architecture')
  if args.model_path is None:
    # Create a model
    print('===>No trained model')
    rnn_model = GRU(input_size=n_input, output_size=n_output)
  else:
    print('===>Load trained model')
    rnn_model = GRU(input_size=n_input_, output_size=n_output)
    rnn_model.load_state_dict(pt.load(args.model_path))
  rnn_model = rnn_model.to(device)
  print(rnn_model)

  # Define optimizer parameters
  learning_rate = 0.01
  optimizer = pt.optim.Adam(rnn_model.parameters(), lr=learning_rate)
  decay_rate = 0.96
  lr_scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)
  decay_cycle = int(len(trajectory_train_dataloader)/10)
  # Log metrics with wandb
  wandb.watch(rnn_model)

  hidden = rnn_model.initHidden(batch_size=args.batch_size)
  cell_state = rnn_model.initCellState(batch_size=args.batch_size)
  # Training a model iterate over dataloader to get each batch and pass to train function
  for batch_idx, batch_train in enumerate(trajectory_train_dataloader):
    if batch_idx == 0:
      continue
    # Training set (Each index in batch came from the collate_fn_padd)
    input_trajectory_train = batch_train['input'][0].to(device)
    input_trajectory_train_lengths = batch_train['input'][1].to(device)
    input_trajectory_train_mask = batch_train['input'][2].to(device)
    input_trajectory_train_startpos = batch_train['input'][3].to(device)
    output_trajectory_train = batch_train['output'][0].to(device)
    output_trajectory_train_lengths = batch_train['output'][1].to(device)
    output_trajectory_train_mask = batch_train['output'][2].to(device)
    output_trajectory_train_startpos = batch_train['output'][3].to(device)
    output_trajectory_train_xyz = batch_train['output'][4].to(device)

    # Validation set
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

    # Log the learning rate
    for param_group in optimizer.param_groups:
      print("[#]Learning rate : ", param_group['lr'])
      wandb.log({'Learning Rate':param_group['lr']})

    # Call function to train
    min_val_loss, hidden, cell_state = train(output_trajectory_train=output_trajectory_train, output_trajectory_train_mask=output_trajectory_train_mask, output_trajectory_train_lengths=output_trajectory_train_lengths, output_trajectory_train_startpos=output_trajectory_train_startpos, output_trajectory_train_xyz=output_trajectory_train_xyz,
                                             input_trajectory_train=input_trajectory_train, input_trajectory_train_mask = input_trajectory_train_mask, input_trajectory_train_lengths=input_trajectory_train_lengths, input_trajectory_train_startpos=input_trajectory_train_startpos,
                                             output_trajectory_val=output_trajectory_val, output_trajectory_val_mask=output_trajectory_val_mask, output_trajectory_val_lengths=output_trajectory_val_lengths, output_trajectory_val_startpos=output_trajectory_val_startpos, output_trajectory_val_xyz=output_trajectory_val_xyz,
                                             input_trajectory_val=input_trajectory_val, input_trajectory_val_mask=input_trajectory_val_mask, input_trajectory_val_lengths=input_trajectory_val_lengths, input_trajectory_val_startpos=input_trajectory_val_startpos,
                                             model=rnn_model, hidden=hidden, cell_state=cell_state, visualize_trajectory_flag=args.visualize_trajectory_flag,
                                             min_val_loss=min_val_loss, model_checkpoint_path=model_checkpoint_path, visualization_path=args.visualization_path)

    if batch_idx % decay_cycle==0 and batch_idx!=0:
      # Decrease learning rate every batch_idx % decay_cycle batch
      for param_group in optimizer.param_groups:
        print("Learning rate : ", param_group['lr'])
      lr_scheduler.step()

  print("[#] Done")
