from __future__ import print_function
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import glob
import os
import argparse
from tqdm import tqdm
from rnn_model import RNN
from lstm_model import LSTM
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from dataloader import TrajectoryDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import PIL.Image
import io
import wandb
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from mpl_toolkits import mplot3d

def visualize_trajectory(output, trajectory_gt, trajectory_startpos, mask, fig=None, flag='Train', n_vis=5):
  # marker_dict for contain the marker properties
  marker_dict_gt = dict(color='rgba(0, 0, 255, 0.4)')
  marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)')
  # Stack the start position to the top of each trajectory
  output = pt.stack([pt.cat([trajectory_startpos[i], output[i]]) for i in range(trajectory_startpos.shape[0])])
  trajectory_gt = pt.stack([pt.cat([trajectory_startpos[i], trajectory_gt[i]]) for i in range(trajectory_startpos.shape[0])])
  # Apply the cummulative summation to get the x, y, z coordinate from the displacement
  output = pt.cumsum(output, dim=1).cpu().detach().numpy()
  trajectory_gt = pt.cumsum(trajectory_gt, dim=1).cpu().detach().numpy()
  # Random the index the be visualize
  vis_idx = np.random.randint(low=0, high=trajectory_startpos.shape[0], size=(n_vis))
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag=='Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter3d(x=output[i][:, 0], y=output[i][:, 1], z=output[i][:, 2], mode='markers', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}".format(flag, i, MSELoss(pt.tensor(output[i]).to(device), pt.tensor(trajectory_gt[i]).to(device), mask=pt.cat((pt.ones((1, 3), dtype=bool).to(device), mask[i]), axis=0)))), row=idx+1, col=col)
    fig.add_trace(go.Scatter3d(x=trajectory_gt[i][:, 0], y=trajectory_gt[i][:, 1], z=trajectory_gt[i][:, 2], mode='markers', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+1, col=col)

  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  fig.update_layout(height=1920, width=1080, margin=dict(l=0, r=0, b=5,t=5,pad=1))
  plotly.offline.plot(fig, filename='./trajectory_visualization.html', auto_open=False)
  wandb.log({"Trajectory Visualization(Col1=Train, Col2=Val)":wandb.Html(open('./trajectory_visualization.html'))})

def MSELoss(output, trajectory_gt, mask, lengths=None, delmask=True):
  mse_loss = pt.sum((((trajectory_gt - output)*10)**2) * mask) / pt.sum(mask)
  return mse_loss

def train(output_trajectory_train, output_trajectory_train_mask, output_trajectory_train_lengths, output_trajectory_train_startpos, input_trajectory_train, input_trajectory_train_mask, input_trajectory_train_lengths, input_trajectory_train_startpos, model, output_trajectory_val, output_trajectory_val_mask, output_trajectory_val_lengths, output_trajectory_val_startpos, input_trajectory_val, input_trajectory_val_mask, input_trajectory_val_lengths, input_trajectory_val_startpos, hidden, cell_state, visualize_trajectory_flag=True, writer=None, min_val_loss=2e10, model_checkpoint_path='./model/'):
  # Training RNN/LSTM model 
  # Run over each example
  # trajectory_train = trajectory_train path with shape (n_trajectory_train, 2) ===> All 2 features are (x0, y0) ... (xn, yn) ;until yn == 0
  # initial_condition_train = Initial conditon with shape (n_trajectory_train, 6) ===> All 6 features are (x, y, angle, velocity, g, timestep)
  # Define models parameters
  learning_rate = 0.001
  n_epochs = 300
  optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)
  # Initial hidden layer for the first RNN Cell
  model.train()
  # Train a model
  for epoch in range(1, n_epochs+1):
    optimizer.zero_grad() # Clear existing gradients from previous epoch
    # Forward PASSING
    # Forward pass for training a model
    output_train, (hidden, cell_state) = model(input_trajectory_train, hidden, cell_state, lengths=input_trajectory_train_lengths)
    # Forward pass for validate a model
    output_val, (_, _) = model(input_trajectory_val, hidden, cell_state, lengths=input_trajectory_val_lengths)
    # Detach for use hidden as a weights in next batch
    cell_state.detach()
    cell_state = cell_state.detach()
    hidden.detach()
    hidden = hidden.detach()
    # Calculate loss of displacement
    train_loss = MSELoss(output=output_train, trajectory_gt=output_trajectory_train, mask=output_trajectory_train_mask, lengths=output_trajectory_train_lengths)
    val_loss = MSELoss(output=output_val, trajectory_gt=output_trajectory_val, mask=output_trajectory_val_mask, lengths=output_trajectory_val_lengths)

    train_loss.backward() # Perform a backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly to the gradients

    if epoch%10 == 0:
      print('Epoch : {}/{}.........'.format(epoch, n_epochs), end='')
      print('Train Loss : {:.3f}'.format(train_loss.item()), end=', ')
      print('Val Loss : {:.3f}'.format(val_loss.item()))
      wandb.log({'Train Loss':train_loss.item(), 'Validation Loss':val_loss.item()})
    if epoch%50 == 0:
      if visualize_trajectory_flag == True:
        # Visualize by make a subplots of trajectory
        n_vis = 5
        fig = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]]*n_vis)
        # Append the start position and apply cummulative summation for transfer the displacement to the x, y, z coordinate. These will done by visualize_trajectory function
        visualize_trajectory(output=pt.mul(output_train, output_trajectory_train_mask), trajectory_gt=output_trajectory_train, trajectory_startpos=output_trajectory_train_startpos, mask=output_trajectory_train_mask, fig=fig, flag='Train', n_vis=n_vis)
        visualize_trajectory(output=pt.mul(output_val, output_trajectory_val_mask), trajectory_gt=output_trajectory_val, trajectory_startpos=output_trajectory_val_startpos, mask=output_trajectory_val_mask, fig=fig, flag='Validation', n_vis=n_vis)
      # Save model checkpoint
      if min_val_loss > val_loss:
        print('[#]Saving a model checkpoint')
        min_val_loss = val_loss
        # Save to directory
        pt.save(model.state_dict(), args.model_checkpoint_path)
        # Save to wandb
        pt.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

  return min_val_loss, hidden, cell_state

def collate_fn_padd(batch):
    '''
    Padding batch of variable length
    '''
    ## Get sequence lengths
    lengths = pt.tensor([trajectory[1:, :].shape[0] for trajectory in batch])
    # Input features : columns 4-5 contain u, v in screen space
    ## Padding 
    input_batch = [pt.Tensor(trajectory[1:, 4:6]) for trajectory in batch]
    input_batch = pt.nn.utils.rnn.pad_sequence(input_batch, batch_first=True)
    ## Retrieve initial position
    input_startpos = pt.stack([pt.Tensor(trajectory[0, 4:6]) for trajectory in batch])
    input_startpos = pt.unsqueeze(input_startpos, dim=1)
    ## Compute mask
    input_mask = (input_batch != 0)

    # Output features : columns 0-2 cotain x, y, z in world space
    ## Padding
    output_batch = [pt.Tensor(trajectory[1:, :3]) for trajectory in batch]
    output_batch = pt.nn.utils.rnn.pad_sequence(output_batch, batch_first=True)
    ## Retrieve initial position
    output_startpos = pt.stack([pt.Tensor(trajectory[0, :3]) for trajectory in batch])
    output_startpos = pt.unsqueeze(output_startpos, dim=1)
    ## Compute mask
    output_mask = (output_batch != 0)

    # print("Mask shape : ", mask.shape)
    return {'input':[input_batch, lengths, input_mask, input_startpos],
            'output':[output_batch, lengths, output_mask, output_startpos]}

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
  parser.add_argument('--wandb_name', dest='wandb_name', type=str, help='WanDB session name', default=None)
  parser.add_argument('--wandb_tags', dest='wandb_tags', type=str, help='WanDB tags name', default=None)
  args = parser.parse_args()

  # Init wandb
  wandb.init(project="ball-trajectory-estimation", name=args.wandb_name, tags=args.wandb_tags)

  # GPU initialization
  if pt.cuda.is_available():
    device = pt.device('cuda')
    print('[%]GPU Enabled')
  else:
    device = pt.device('cpu')
    print('[%]GPU Disabled, CPU Enabled')

  # Initial writer for tensorboard
  writer = SummaryWriter('trajectory_tensorboard/{}'.format(args.dataset_train_path))

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
    unpacked = pt.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
    print("Unpacked equality : ", pt.eq(batch['input'][0], unpacked[0]).all())
    print("===============================================================================================================================================================")

  # Model definition
  hidden_dim = 64
  n_output = 3 # Contain the depth information of the trajectory
  n_input = 2 # Contain following this trajectory parameters (u, v) position from tracking
  min_val_loss = 2e10
  print('[#]Model Architecture')
  if args.model_path is None:
    # Create a model
    print('===>No trained model')
    rnn_model = LSTM(input_size=n_input, output_size=n_output, hidden_dim=hidden_dim, n_layers=8)
  else:
    print('===>Load trained model')
    rnn_model = LSTM(input_size=n_input_, output_size=n_output, hidden_dim=hidden_dim, n_layers=8)
    rnn_model.load_state_dict(pt.load(args.model_path))
  rnn_model = rnn_model.to(device)
  print(rnn_model)
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

    # Call function to train
    min_val_loss, hidden, cell_state = train(output_trajectory_train=output_trajectory_train, output_trajectory_train_mask=output_trajectory_train_mask, output_trajectory_train_lengths=output_trajectory_train_lengths, output_trajectory_train_startpos=output_trajectory_train_startpos,
                                             input_trajectory_train=input_trajectory_train, input_trajectory_train_mask = input_trajectory_train_mask, input_trajectory_train_lengths=input_trajectory_train_lengths, input_trajectory_train_startpos=input_trajectory_train_startpos,
                                             output_trajectory_val=output_trajectory_val, output_trajectory_val_mask=output_trajectory_val_mask, output_trajectory_val_lengths=output_trajectory_val_lengths, output_trajectory_val_startpos=output_trajectory_val_startpos,
                                             input_trajectory_val=input_trajectory_val, input_trajectory_val_mask=input_trajectory_val_mask, input_trajectory_val_lengths=input_trajectory_val_lengths, input_trajectory_val_startpos=input_trajectory_val_startpos,
                                             model=rnn_model, hidden=hidden, cell_state=cell_state, visualize_trajectory_flag=args.visualize_trajectory_flag,
                                             writer=writer, min_val_loss=min_val_loss, model_checkpoint_path=args.model_checkpoint_path)

  print("[#] Done")
