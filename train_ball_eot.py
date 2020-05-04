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
from models.bigru_model import BiGRU
from models.gru_model import GRU

def make_visualize(output_train_eot, output_trajectory_train_startpos, input_trajectory_train_lengths, output_trajectory_train_maks, output_val_eot,output_trajectory_val_startpos, input_trajectory_val_lengths, output_trajectory_val_mask, visualization_path):
  # Visualize by make a subplots of trajectory
  n_vis = 5
  # Random the index the be visualize
  train_vis_idx = np.random.randint(low=0, high=input_trajectory_train_startpos.shape[0], size=(n_vis))
  val_vis_idx = np.random.randint(low=0, high=input_trajectory_val_startpos.shape[0], size=(n_vis))
  fig = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)

  # Visualize the End of trajectory(EOT) flag
  fig_eot = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  visualize_eot(output_eot=output_train_eot.clone(), eot_gt=output_trajectory_train_uv[..., -1], eot_startpos=output_trajectory_train_startpos[..., -1], lengths=input_trajectory_train_lengths, mask=output_trajectory_train_mask[..., -1], fig=fig_eot, flag='Train', n_vis=n_vis, vis_idx=train_vis_idx)
  visualize_eot(output_eot=output_val_eot.clone(), eot_gt=output_trajectory_val_uv[..., -1], eot_startpos=output_trajectory_val_startpos[..., -1], lengths=input_trajectory_val_lengths, mask=output_trajectory_val_mask[..., -1], fig=fig_eot, flag='Validation', n_vis=n_vis, vis_idx=val_vis_idx)
  plotly.offline.plot(fig_eot, filename='./{}/trajectory_visualization_depth_eot_prediction.html'.format(visualization_path), auto_open=False)
  wandb.log({"End Of Trajectory flag Prediction : (Col1=Train, Col2=Val)":wandb.Html(open('./{}/trajectory_visualization_depth_eot_prediction.html'.format(visualization_path)))})

def visualize_eot(output_eot, eot_gt, eot_startpos, lengths, mask, vis_idx, fig=None, flag='train', n_vis=5):
  # Add the feature dimension using unsqueeze
  eot_gt = pt.unsqueeze(eot_gt, dim=2)
  mask = pt.unsqueeze(mask, dim=2)
  eot_startpos = pt.unsqueeze(eot_startpos, dim=2)
  # eot_gt : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output_eot = pt.stack([pt.cat([output_eot[i], eot_startpos[i]]) for i in range(eot_startpos.shape[0])])
  output_eot *= mask
  eot_gt *= mask
  # print(pt.cat((output_eot[0][:lengths[0]+1], pt.sigmoid(output_eot)[0][:lengths[0]+1], eot_gt[0][:lengths[0]+1], ), dim=1))
  output_eot = pt.sigmoid(output_eot)
  pos_weight = pt.sum(eot_gt == 0)/pt.sum(eot_gt==1)
  neg_weight = 1
  eps = 1e-10
  # detach() for visualization
  eot_loss = pt.mean(-((pos_weight * eot_gt * pt.log(output_eot+eps)) + (neg_weight * (1-eot_gt)*pt.log(1-output_eot+eps))), dim=1).cpu().detach().numpy()
  output_eot = output_eot.cpu().detach().numpy()
  eot_gt = eot_gt.cpu().detach().numpy()
  lengths = lengths.cpu().detach().numpy()
  # marker_dict for contain the marker properties
  marker_dict_gt = dict(color='rgba(0, 0, 255, .7)', size=5)
  marker_dict_pred = dict(color='rgba(255, 0, 0, .7)', size=5)
  # Random the index the be visualize
  vis_idx = np.random.randint(low=0, high=eot_startpos.shape[0], size=(n_vis))
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]+1).reshape(-1,), y=output_eot[i][:lengths[i]+1, :].reshape(-1,), mode='markers+lines', marker=marker_dict_pred, name="{}-EOT Predicted [{}], EOTLoss = {:.3f}".format(flag, i, eot_loss[i][0])), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]+1).reshape(-1,), y=eot_gt[i][:lengths[i]+1, :].reshape(-1,), mode='markers+lines', marker=marker_dict_gt, name="{}-Ground Truth EOT [{}]".format(flag, i)), row=idx+1, col=col)

def EndOfTrajectoryLoss(output_eot, eot_gt, eot_startpos, mask, lengths):
  # Add the feature dimension using unsqueeze
  eot_gt = pt.unsqueeze(eot_gt, dim=2)
  mask = pt.unsqueeze(mask, dim=2)
  eot_startpos = pt.unsqueeze(eot_startpos, dim=2)
  # eot_gt : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output_eot = pt.stack([pt.cat([output_eot[i], eot_startpos[i]]) for i in range(eot_startpos.shape[0])])
  output_eot *= mask
  eot_gt *= mask
  # Concat the startpos of end_of_trajectory
  # print("EndOfTrajectoryLoss function : ")
  # print(output_eot.shape, eot_gt.shape, mask.shape, lengths.shape, eot_startpos.shape)
  # print((output_eot * mask)[0][:lengths[0]+1].shape)
  # print((eot_gt * mask)[0][:lengths[0]+1].shape)
  # BCELoss
  # eot_loss = pt.nn.BCELoss(reduction='mean')(pt.sigmoid(output_eot) + 1e-10, eot_gt)
  # print("EOT Loss : ", eot_loss)
  # BCEWithLogitsLoss
  # eot_loss = (1/args.batch_size) * (pt.sum(pt.tensor([pt.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pt.ones_like(eot_gt[i][:lengths[i]+1])*100)(output_eot[i][:lengths[i]+1], eot_gt[i][:lengths[i]+1]) for i in range(eot_gt.shape[0])], requires_grad=True)))
  # Implement from scratch
  # print(pt.cat((output_eot[0][:lengths[0]+1], pt.sigmoid(output_eot)[0][:lengths[0]+1], eot_gt[0][:lengths[0]+1], ), dim=1))
  eot_gt = pt.cat(([eot_gt[i][:lengths[i]+1] for i in range(args.batch_size)]))
  # print(eot_gt)
  output_eot = pt.sigmoid(pt.cat(([output_eot[i][:lengths[i]+1] for i in range(args.batch_size)])))
  # print(output_eot)
  # print((eot_gt * pt.log(output_eot))[:250])
  # print(((1-eot_gt)*pt.log(1-output_eot))[:250])
  # exit()
  # print(eot_gt[:200])
  # print(output_eot[:200])
  pos_weight = pt.sum(eot_gt == 0)/pt.sum(eot_gt==1)
  neg_weight = 1
  # Prevent of pt.log(-value)
  eps = 1e-10
  eot_loss = pt.mean(-((pos_weight * eot_gt * pt.log(output_eot + eps)) + (neg_weight * (1-eot_gt)*pt.log(1-output_eot + eps))))
  # eot_loss = pt.mean(((eot_gt - output_eot)**2))
  return eot_loss * 100

def train(output_trajectory_train, output_trajectory_train_mask, output_trajectory_train_lengths, output_trajectory_train_startpos, output_trajectory_train_uv, input_trajectory_train, input_trajectory_train_mask, input_trajectory_train_lengths, input_trajectory_train_startpos, model, output_trajectory_val, output_trajectory_val_mask, output_trajectory_val_lengths, output_trajectory_val_startpos, output_trajectory_val_uv, input_trajectory_val, input_trajectory_val_mask, input_trajectory_val_lengths, input_trajectory_val_startpos, hidden, cell_state, optimizer, visualize_trajectory_flag=True, min_val_loss=2e10, model_checkpoint_path='./model/', visualization_path='./visualize_html/'):
  # Training RNN/LSTM model
  # Run over each example
  n_epochs = 500
  # Initial hidden layer for the first RNN Cell
  # Train a model
  for epoch in range(1, n_epochs+1):
    hidden = rnn_model.initHidden(batch_size=args.batch_size)
    cell_state = rnn_model.initCellState(batch_size=args.batch_size)
    # Training mode
    model.train()
    optimizer.zero_grad() # Clear existing gradients from previous epoch
    # Forward PASSING
    # Forward pass for training a model  
    output_train_eot, (_, _) = model(input_trajectory_train, hidden, cell_state, lengths=input_trajectory_train_lengths)
    # Evaluating mode
    model.eval()
    # Forward pass for validate a model
    output_val_eot, (_, _) = model(input_trajectory_val, hidden, cell_state, lengths=input_trajectory_val_lengths)
    # Detach for use hidden as a weights in next batch
    cell_state.detach()
    cell_state = cell_state.detach()
    hidden.detach()
    hidden = hidden.detach()

    # Calculate loss of unprojected trajectory
    train_eot_loss = EndOfTrajectoryLoss(output_eot=output_train_eot.clone(), eot_gt=output_trajectory_train_uv[..., -1], mask=output_trajectory_train_mask[..., -1], lengths=output_trajectory_train_lengths, eot_startpos=input_trajectory_train_startpos[..., -1])
    val_eot_loss = EndOfTrajectoryLoss(output_eot=output_val_eot.clone(), eot_gt=output_trajectory_val_uv[..., -1], mask=output_trajectory_val_mask[..., -1], lengths=output_trajectory_val_lengths, eot_startpos=input_trajectory_val_startpos[..., -1])

    train_loss = train_eot_loss
    val_loss = val_eot_loss

    train_loss.backward() # Perform a backpropagation and calculates gradients
    pt.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
    # optimizer.step() # Updates the weights accordingly to the gradients
    if epoch%10 == 0:
      print('Epoch : {}/{}.........'.format(epoch, n_epochs), end='')
      print('Train Loss : {:.3f}'.format(train_loss.item()), end=', ')
      print('Val Loss : {:.3f}'.format(val_loss.item()))
      wandb.log({'Train Loss':train_loss.item(), 'Validation Loss':val_loss.item()})
      if min_val_loss > val_loss:
        # Save model checkpoint
        print('[#]Saving a model checkpoint')
        min_val_loss = val_loss
        # Save to directory
        pt.save(model.state_dict(), model_checkpoint_path)
        # Save to wandb
        pt.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

    if epoch%25 == 0:
      if visualize_trajectory_flag == True:
        make_visualize(output_train_eot=output_train_eot, output_trajectory_train_startpos=output_trajectory_train_startpos, input_trajectory_train_lengths=input_trajectory_train_lengths, output_trajectory_train_maks=output_trajectory_train_mask, output_val_eot=output_val_eot, output_trajectory_val_startpos=output_trajectory_val_startpos, input_trajectory_val_lengths=input_trajectory_val_lengths, output_trajectory_val_mask=output_trajectory_val_mask, visualization_path=visualization_path)

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
    input_batch = [pt.Tensor(trajectory[1:, [4, 5]]) for trajectory in batch] # (4, 5, -1) = (u, v)
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=-1)
    ## Retrieve initial position (u, v, depth)
    input_startpos = pt.stack([pt.Tensor(trajectory[0, [4, 5, -2]]) for trajectory in batch])  # (4, 5, 6, -2) = (u, v, end_of_trajectory)
    input_startpos = pt.unsqueeze(input_startpos, dim=1)
    ## Compute mask
    input_mask = (input_batch != -1)

    # Output features : columns 6 cotain depth from camera to projected screen
    ## Padding
    output_batch = [pt.Tensor(trajectory[1:, -2]) for trajectory in batch]
    output_batch = pad_sequence(output_batch, batch_first=True)
    ## Retrieve initial position
    output_startpos = pt.stack([pt.Tensor(trajectory[0, [-2]]) for trajectory in batch])
    output_startpos = pt.unsqueeze(output_startpos, dim=1)
    ## Retrieve the x, y, z in world space for compute the reprojection error (x', y', z' <===> x, y, z)
    output_uv = [pt.Tensor(trajectory[:, [4, 5, -2]]) for trajectory in batch]
    output_uv = pad_sequence(output_uv, batch_first=True, padding_value=-1)
    ## Compute mask
    output_mask = (output_uv != -1)
    ## Compute cummulative summation to form a trajectory from displacement every columns except the end_of_trajectory
    # print(output_xyz[..., :-1].shape, pt.unsqueeze(output_xyz[..., -1], dim=2).shape)
    output_uv = pt.cat((pt.cumsum(output_uv[..., :-1], dim=1), pt.unsqueeze(output_uv[..., -1], dim=2)), dim=2)

    # print("Mask shape : ", mask.shape)
    return {'input':[input_batch, lengths, input_mask, input_startpos],
            'output':[output_batch, lengths, output_mask, output_startpos, output_uv]}

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
  n_output = 1 # Contain the depth information of the trajectory and the end_of_trajectory flag
  n_input = 2 # Contain following this trajectory parameters (u, v, end_of_trajectory) position from tracking
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
  decay_cycle = int(len(trajectory_train_dataloader)/10)
  # Log metrics with wandb
  wandb.watch(rnn_model)

  hidden = rnn_model.initHidden(batch_size=args.batch_size)
  cell_state = rnn_model.initCellState(batch_size=args.batch_size)
  # Training a model iterate over dataloader to get each batch and pass to train function
  for batch_idx, batch_train in enumerate(trajectory_train_dataloader):
    # Training set (Each index in batch_train came from the collate_fn_padd)
    input_trajectory_train = batch_train['input'][0].to(device)
    input_trajectory_train_lengths = batch_train['input'][1].to(device)
    input_trajectory_train_mask = batch_train['input'][2].to(device)
    input_trajectory_train_startpos = batch_train['input'][3].to(device)
    output_trajectory_train = batch_train['output'][0].to(device)
    output_trajectory_train_lengths = batch_train['output'][1].to(device)
    output_trajectory_train_mask = batch_train['output'][2].to(device)
    output_trajectory_train_startpos = batch_train['output'][3].to(device)
    output_trajectory_train_uv = batch_train['output'][4].to(device)

    # Validation set (Get each batch for each training iteration
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
    output_trajectory_val_uv = batch_val['output'][4].to(device)

    # Log the learning rate
    for param_group in optimizer.param_groups:
      print("[#]Learning rate : ", param_group['lr'])
      wandb.log({'Learning Rate':param_group['lr']})

    # Call function to train
    min_val_loss, hidden, cell_state = train(output_trajectory_train=output_trajectory_train, output_trajectory_train_mask=output_trajectory_train_mask,
                                             output_trajectory_train_lengths=output_trajectory_train_lengths, output_trajectory_train_startpos=output_trajectory_train_startpos, output_trajectory_train_uv=output_trajectory_train_uv,
                                             input_trajectory_train=input_trajectory_train, input_trajectory_train_mask = input_trajectory_train_mask,
                                             input_trajectory_train_lengths=input_trajectory_train_lengths, input_trajectory_train_startpos=input_trajectory_train_startpos,
                                             output_trajectory_val=output_trajectory_val, output_trajectory_val_mask=output_trajectory_val_mask,
                                             output_trajectory_val_lengths=output_trajectory_val_lengths, output_trajectory_val_startpos=output_trajectory_val_startpos,
                                             input_trajectory_val=input_trajectory_val, input_trajectory_val_mask=input_trajectory_val_mask, output_trajectory_val_uv=output_trajectory_val_uv,
                                             input_trajectory_val_lengths=input_trajectory_val_lengths, input_trajectory_val_startpos=input_trajectory_val_startpos,
                                             model=rnn_model, hidden=hidden, cell_state=cell_state, visualize_trajectory_flag=args.visualize_trajectory_flag,
                                             min_val_loss=min_val_loss, model_checkpoint_path=model_checkpoint_path, optimizer=optimizer)

    if batch_idx % decay_cycle==0 and batch_idx!=0:
      # Decrease learning rate every batch_idx % decay_cycle batch
      for param_group in optimizer.param_groups:
        print("Learning rate : ", param_group['lr'])
      lr_scheduler.step()

  print("[#] Done")
