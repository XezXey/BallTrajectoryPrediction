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
import json

def visualize_trajectory(output, trajectory_gt, trajectory_startpos, lengths, writer, flag='predict', n_vis=5):
  output = output.cpu().detach().numpy()
  trajectory_gt = trajectory_gt.cpu().detach().numpy()
  # Random the index the be visualize
  vis_idx = np.random.randint(low=0, high=trajectory_startpos.shape[0], size=(n_vis))
  # Visualize by make a subplots of trajectory
  fig = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter'}]]*n_vis)
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter3d(x=output[i][:lengths[i]+1, 0], y=output[i][:lengths[i]+1, 1], z=output[i][:lengths[i]+1, 2], mode='markers', name="Estimated Trajectory [{}]".format(i)), row=idx+1, col=1)
    fig.add_trace(go.Scatter3d(x=trajectory_gt[i][:lengths[i]+1, 0], y=trajectory_gt[i][:lengths[i]+1, 1], z=trajectory_gt[i][:lengths[i]+1, 2], mode='markers', name="Ground Truth Trajectory [{}]".format(i)), row=idx+1, col=1)

  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  fig.update_layout(height=1920, width=1080, margin=dict(l=0, r=0, b=5,t=5,pad=1))
  plotly.offline.plot(fig, filename='./trajectory_visualization.html', auto_open=False)
  wandb.log({"Trajectory Visualization":wandb.Html(open('./trajectory_visualization.html'))})

def MSELoss(output, trajectory_gt, mask, delmask=True):
  mse_loss = pt.sum(((trajectory_gt - output)**2) * mask) / pt.sum(mask)
  return mse_loss

def projectToWorldSpace(screen_space, depth, projection_matrix, camera_to_world_matrix):
  depth = depth.view(-1)
  screen_width = 1920.
  screen_height = 1080.
  screen_space = pt.div(screen_space, pt.tensor([screen_width, screen_height]).to(device)) # Normalize : (width, height) -> (-1, 1)
  screen_space = (screen_space * 2.0) - pt.ones(size=(screen_space.size()), dtype=pt.float32).to(device) # Normalize : (width, height) -> (-1, 1)
  screen_space = (screen_space.t() * depth).t()   # Normalize : (-1, 1) -> (-depth, depth)
  screen_space = pt.stack((screen_space[:, 0], screen_space[:, 1], depth, pt.ones(depth.shape[0], dtype=pt.float32).to(device)), axis=1) # Stack the screen with depth and w ===> (x, y, depth, 1)
  screen_space = ((camera_to_world_matrix @ projection_matrix) @ screen_space.t()).t() # Reprojected
  return screen_space[:, :3]

def train(output_trajectory_train, output_trajectory_train_mask, output_trajectory_train_lengths, output_trajectory_train_startpos, output_trajectory_train_xyz, input_trajectory_train, input_trajectory_train_mask, input_trajectory_train_lengths, input_trajectory_train_startpos, model, output_trajectory_val, output_trajectory_val_mask, output_trajectory_val_lengths, output_trajectory_val_startpos, output_trajectory_val_xyz, input_trajectory_val, input_trajectory_val_mask, input_trajectory_val_lengths, input_trajectory_val_startpos, hidden, cell_state, projection_matrix, camera_to_world_matrix, visualize_trajectory_flag=True, writer=None, min_val_loss=2e10, model_checkpoint_path='./model/'):
  # Training RNN/LSTM model 
  # Run over each example
  # trajectory_train = trajectory_train path with shape (n_trajectory_train, 2) ===> All 2 features are (x0, y0) ... (xn, yn) ;until yn == 0
  # initial_condition_train = Initial conditon with shape (n_trajectory_train, 6) ===> All 6 features are (x, y, angle, velocity, g, timestep)
  # Define models parameters
  learning_rate = 0.001
  n_epochs = 500
  optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)
  # Initial hidden layer for the first RNN Cell
  model.train()
  # Train a model
  for epoch in range(1, n_epochs+1):
    optimizer.zero_grad() # Clear existing gradients from previous epoch
    # Forward PASSING
    # Forward pass for training a model  
    output_train, (hidden, cell_state) = model(input_trajectory_train, hidden, cell_state, lengths=input_trajectory_train_lengths)
    # (This step we get the displacement of depth by input the displacement of u and v)
    # Apply cummulative summation to output
    input_trajectory_train_temp = pt.stack([pt.cat([input_trajectory_train_startpos[i][:, :2], input_trajectory_train[i].clone().detach()]) for i in range(input_trajectory_train_startpos.shape[0])])
    input_trajectory_train_temp = pt.cumsum(input_trajectory_train_temp, dim=1)
    output_train = pt.stack([pt.cat([input_trajectory_train_startpos[i][:, -1].view(-1, 1), output_train[i]]) for i in range(input_trajectory_train_startpos.shape[0])])
    output_train = pt.cumsum(output_train, dim=1)
    # Project the (u, v, depth) to world space
    output_train = pt.stack([projectToWorldSpace(screen_space=input_trajectory_train_temp[i], depth=output_train[i], projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix) for i in range(output_train.shape[0])])

    # Forward pass for validate a model
    output_val, (_, _) = model(input_trajectory_val, hidden, cell_state, lengths=input_trajectory_val_lengths)
    # (This step we get the displacement of depth by input the displacement of u and v)
    # Apply cummulative summation to output
    input_trajectory_val_temp = pt.stack([pt.cat([input_trajectory_val_startpos[i][:, :2], input_trajectory_val[i].clone().detach()]) for i in range(input_trajectory_val_startpos.shape[0])])
    input_trajectory_val_temp = pt.cumsum(input_trajectory_val_temp, dim=1)
    output_val = pt.stack([pt.cat([input_trajectory_val_startpos[i][:, -1].view(-1, 1), output_val[i]]) for i in range(input_trajectory_val_startpos.shape[0])])
    output_val = pt.cumsum(output_val, dim=1)
    # Project the (u, v, depth) to world space
    output_val = pt.stack([projectToWorldSpace(screen_space=input_trajectory_val_temp[i], depth=output_val[i], projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix) for i in range(output_val.shape[0])])
    # Detach for use hidden as a weights in next batch
    cell_state.detach()
    cell_state = cell_state.detach()
    hidden.detach()
    hidden = hidden.detach()

    # Calculate loss of displacement
    train_loss = MSELoss(output=output_train, trajectory_gt=output_trajectory_train_xyz, mask=output_trajectory_train_mask)
    val_loss = MSELoss(output=output_val, trajectory_gt=output_trajectory_val_xyz, mask=output_trajectory_val_mask)

    train_loss.backward() # Perform a backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly to the gradients
    if epoch%10 == 0:
      print('Epoch : {}/{}.........'.format(epoch, n_epochs), end='')
      print('Train Loss : {:.3f}'.format(train_loss.item()), end=', ')
      print('Val Loss : {:.3f}'.format(val_loss.item()))
      wandb.log({'Train Loss':train_loss.item(), 'Validation Loss':val_loss.item()})

    if epoch%50 == 0:
      writer.add_scalars('Loss/', {'Training loss':train_loss.item(),
                                  'Validation loss':val_loss.item()}, epoch)
      if visualize_trajectory_flag == True:
        # Append the start position and apply cummulative summation for transfer the displacement to the x, y, z coordinate. These will done by visualize_trajectory function
        visualize_trajectory(output=pt.mul(output_train, output_trajectory_train_mask), trajectory_gt=output_trajectory_train_xyz, trajectory_startpos=output_trajectory_train_startpos, lengths=input_trajectory_train_lengths, writer=writer, flag='train')
        visualize_trajectory(output=pt.mul(output_val, output_trajectory_val_mask), trajectory_gt=output_trajectory_val_xyz, trajectory_startpos=output_trajectory_val_startpos, lengths=input_trajectory_val_lengths, writer=writer, flag='train')
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
    input_batch = pad_sequence(input_batch, batch_first=True)
    ## Retrieve initial position
    input_startpos = pt.stack([pt.Tensor(trajectory[0, 4:7]) for trajectory in batch])
    input_startpos = pt.unsqueeze(input_startpos, dim=1)
    ## Compute mask
    input_mask = (input_batch != 0)

    # Output features : columns 6 cotain depth from camera to projected screen
    ## Padding
    output_batch = [pt.Tensor(trajectory[:, 6]) for trajectory in batch]
    output_batch = pad_sequence(output_batch, batch_first=True)
    ## Retrieve initial position
    output_startpos = pt.stack([pt.Tensor(trajectory[0, :3]) for trajectory in batch])
    output_startpos = pt.unsqueeze(output_startpos, dim=1)
    ## Retrieve the x, y, z in world space for compute the reprojection error (x', y', z' <===> x, y, z)
    output_xyz = [pt.Tensor(trajectory[:, :3]) for trajectory in batch]
    output_xyz = pad_sequence(output_xyz, batch_first=True)
    ## Compute mask
    output_mask = (output_xyz != 0)
    ## Compute cummulative summation to form a trajectory from displacement
    output_xyz = pt.cumsum(output_xyz, dim=1)

    # print("Mask shape : ", mask.shape)
    return {'input':[input_batch, lengths, input_mask, input_startpos],
            'output':[output_batch, lengths, output_mask, output_startpos, output_xyz]}

if __name__ == '__main__':
  print('[#]Training : Trajectory Estimation')
  # Init wandb
  wandb.init(project="ball-trajectory-estimation")
  # Argumentparser for input
  parser = argparse.ArgumentParser(description='Predict the 2D projectile')
  parser.add_argument('--dataset_path', dest='dataset_path', type=str, help='Path to dataset', required=True)
  parser.add_argument('--batch_size', dest='batch_size', type=int, help='Size of batch', default=50)
  parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, help='Type of trajectory(Rolling, Projectile, MagnusProjectile)', default='Projectile')
  parser.add_argument('--visualize_trajectory_flag', dest='visualize_trajectory_flag', type=bool, help='Visualize the trajectory', default=False)
  parser.add_argument('--model_checkpoint_path', dest='model_checkpoint_path', type=str, help='Path to save a model checkpoint', required=True)
  parser.add_argument('--model_path', dest='model_path', type=str, help='Path to load a trained model checkpoint', default=None)
  parser.add_argument('--cam_params_file', dest='cam_params_file', type=str, help='Path to camera parameters file(Intrinsic/Extrinsic)')
  args = parser.parse_args()

  # GPU initialization
  if pt.cuda.is_available():
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

  # Initial writer for tensorboard
  writer = SummaryWriter('trajectory_tensorboard/{}'.format(args.dataset_path))

  # Create Datasetloader
  trajectory_dataset = TrajectoryDataset(dataset_path=args.dataset_path, trajectory_type=args.trajectory_type)
  trajectory_dataloader = DataLoader(trajectory_dataset, batch_size=args.batch_size, num_workers=10, shuffle=False, collate_fn=collate_fn_padd, pin_memory=True, drop_last=True)

  # Dataset format
  # Trajectory path : (x0, y0) ... (xn, yn)
  print("======================================================Summary Batch (batch_size = {})=========================================================================".format(args.batch_size))
  for key, batch in enumerate(trajectory_dataloader):
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
  n_output = 1 # Contain the depth information of the trajectory
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
  for batch_idx, batch in enumerate(trajectory_dataloader):
    if batch_idx == 0:
      continue
    # Training set (Each index in batch came from the collate_fn_padd)
    input_trajectory_train = batch['input'][0].to(device)
    input_trajectory_train_lengths = batch['input'][1].to(device)
    input_trajectory_train_mask = batch['input'][2].to(device)
    input_trajectory_train_startpos = batch['input'][3].to(device)
    output_trajectory_train = batch['output'][0].to(device)
    output_trajectory_train_lengths = batch['output'][1].to(device)
    output_trajectory_train_mask = batch['output'][2].to(device)
    output_trajectory_train_startpos = batch['output'][3].to(device)
    output_trajectory_train_xyz = batch['output'][4].to(device)
    # Validation set
    input_trajectory_val = batch['input'][0].to(device)
    input_trajectory_val_lengths = batch['input'][1].to(device)
    input_trajectory_val_mask = batch['input'][2].to(device)
    input_trajectory_val_startpos = batch['input'][3].to(device)
    output_trajectory_val = batch['output'][0].to(device)
    output_trajectory_val_lengths = batch['output'][1].to(device)
    output_trajectory_val_mask = batch['output'][2].to(device)
    output_trajectory_val_startpos = batch['output'][3].to(device)
    output_trajectory_val_xyz = batch['output'][4].to(device)

    # Call function to train
    min_val_loss, hidden, cell_state = train(output_trajectory_train=output_trajectory_train, output_trajectory_train_mask=output_trajectory_train_mask,
                                             output_trajectory_train_lengths=output_trajectory_train_lengths, output_trajectory_train_startpos=output_trajectory_train_startpos, output_trajectory_train_xyz=output_trajectory_train_xyz,
                                             input_trajectory_train=input_trajectory_train, input_trajectory_train_mask = input_trajectory_train_mask,
                                             input_trajectory_train_lengths=input_trajectory_train_lengths, input_trajectory_train_startpos=input_trajectory_train_startpos,
                                             output_trajectory_val=output_trajectory_val, output_trajectory_val_mask=output_trajectory_val_mask,
                                             output_trajectory_val_lengths=output_trajectory_val_mask, output_trajectory_val_startpos=output_trajectory_val_startpos,
                                             input_trajectory_val=input_trajectory_val, input_trajectory_val_mask=input_trajectory_val_mask, output_trajectory_val_xyz=output_trajectory_val_xyz,
                                             input_trajectory_val_lengths=input_trajectory_val_lengths, input_trajectory_val_startpos=input_trajectory_val_startpos,
                                             model=rnn_model, hidden=hidden, cell_state=cell_state, visualize_trajectory_flag=args.visualize_trajectory_flag,
                                             writer=writer, min_val_loss=min_val_loss, model_checkpoint_path=args.model_checkpoint_path, projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix)

  print("[#] Done")