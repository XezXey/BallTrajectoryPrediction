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
from sklearn.metrics import confusion_matrix
# Dataloader
from utils.dataloader import TrajectoryDataset
# Models
from models.rnn_model import RNN
from models.lstm_model import LSTM
from models.bilstm_model import BiLSTM
from models.bigru_model import BiGRU
from models.bigru_model_residual_list import BiGRUResidualList
from models.bigru_model_residual_add import BiGRUResidualAdd
from models.gru_model import GRU

def make_visualize(output_train_eot, output_trajectory_train_startpos, input_trajectory_train_lengths, output_trajectory_train_maks, output_val_eot,output_trajectory_val_startpos, input_trajectory_val_lengths, output_trajectory_val_mask, visualization_path, eot_gt_train, eot_gt_val, input_trajectory_train, input_trajectory_val):
  # Visualize by make a subplots of trajectory
  n_vis = 5
  # Random the index the be visualize
  train_vis_idx = np.random.randint(low=0, high=input_trajectory_train_startpos.shape[0], size=(n_vis))
  val_vis_idx = np.random.randint(low=0, high=input_trajectory_val_startpos.shape[0], size=(n_vis))

  # Visualize the displacement
  fig_displacement = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  visualize_displacement(in_f=input_trajectory_train, out_f=output_train_eot, gt=eot_gt_train, mask=input_trajectory_train_mask, lengths=input_trajectory_train_lengths, n_vis=n_vis, vis_idx=train_vis_idx, fig=fig_displacement, flag='Train')
  visualize_displacement(in_f=input_trajectory_val, out_f=output_val_eot, gt=eot_gt_val, mask=input_trajectory_val_mask, lengths=input_trajectory_val_lengths, n_vis=n_vis, vis_idx=val_vis_idx, fig=fig_displacement, flag='Validation')
  wandb.log({"DISPLACEMENT VISUALIZATION":fig_displacement})

  # Visualize the End of trajectory(EOT) flag
  fig_eot = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  visualize_eot(output_eot=output_train_eot.clone(), eot_gt=output_trajectory_train_uv[..., -1], eot_startpos=output_trajectory_train_startpos[..., -1], lengths=input_trajectory_train_lengths, mask=output_trajectory_train_mask[..., -1], fig=fig_eot, flag='Train', n_vis=n_vis, vis_idx=train_vis_idx)
  visualize_eot(output_eot=output_val_eot.clone(), eot_gt=output_trajectory_val_uv[..., -1], eot_startpos=output_trajectory_val_startpos[..., -1], lengths=input_trajectory_val_lengths, mask=output_trajectory_val_mask[..., -1], fig=fig_eot, flag='Validation', n_vis=n_vis, vis_idx=val_vis_idx)
  wandb.log({"End Of Trajectory flag Prediction : (Col1=Train, Col2=Val)":fig_eot})

def visualize_displacement(in_f, out_f, gt, mask, lengths, vis_idx, fig=None, flag='train', n_vis=5):
  threshold = 0.8
  out_f = pt.sigmoid(out_f) > threshold
  in_f = in_f.cpu().detach().numpy()
  out_f = out_f.cpu().detach().numpy()
  lengths = lengths.cpu().detach().numpy()
  gt = pt.unsqueeze(gt, dim=-1).cpu().detach().numpy()
  marker_dict_gt = dict(color='rgba(0, 0, 255, 0.7)', size=3)
  marker_dict_pred = dict(color='rgba(255, 0, 0, 0.7)', size=3)
  marker_dict_eot = dict(color='rgba(150, 255, 125, 0.7)', size=3)
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=in_f[i][:lengths[i]+1, 0], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Displacement of U'.format(flag, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=in_f[i][:lengths[i]+1, 1], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Displacement of V'.format(flag, i)), row=idx+1, col=col)
    # fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=gt[i][:lengths[i]+1, :], mode='markers+lines', marker=marker_dict_eot, name='{}-traj#{}-GroundTruth EOT'.format(flag, i)), row=idx+1, col=col)
    # fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=out_f[i][:lengths[i]+1, :], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-Prediction EOT'.format(flag, i)), row=idx+1, col=col)


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
  # Thresholding the EOT to be class True/False
  threshold = 0.8
  output_eot = output_eot > threshold

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
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]+1).reshape(-1,), y=output_eot[i][:lengths[i]+1, :].reshape(-1,), mode='markers+lines', marker=marker_dict_pred, name="{}-EOT Predicted [{}], EOTLoss = {:.3f}".format(flag, i, eot_loss[i][0])), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]+1).reshape(-1,), y=eot_gt[i][:lengths[i]+1, :].reshape(-1,), mode='markers+lines', marker=marker_dict_gt, name="{}-Ground Truth EOT [{}]".format(flag, i)), row=idx+1, col=col)

def eot_metrics_log(eot_gt, output_eot, lengths, flag):
  output_eot = output_eot > 0.8
  # Output of confusion_matrix.ravel() = [TN, FP ,FN, TP]
  cm_each_trajectory = np.array([confusion_matrix(y_pred=output_eot[i][:lengths[i]+1, :], y_true=eot_gt[i][:lengths[i]+1]).ravel() for i in range(lengths.shape[0])])
  n_accepted_trajectory = np.sum(np.logical_and(cm_each_trajectory[:, 1]==0., cm_each_trajectory[:, 2] == 0.))
  cm_batch = np.sum(cm_each_trajectory, axis=0)
  tn, fp, fn, tp = cm_batch
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1_score = 2 * (precision * recall) / (precision + recall)
  wandb.log({'{} Precision'.format(flag):precision, '{} Recall'.format(flag):recall, '{} F1-score'.format(flag):f1_score, '{}-#N accepted trajectory(Perfect EOT without FN, FP)'.format(flag):n_accepted_trajectory})

def EndOfTrajectoryLoss(output_eot, eot_gt, eot_startpos, mask, lengths, flag='train'):
  # Add the feature dimension using unsqueeze
  eot_gt = pt.unsqueeze(eot_gt, dim=2)
  mask = pt.unsqueeze(mask, dim=2)
  eot_startpos = pt.unsqueeze(eot_startpos, dim=2)
  # output_eot : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output_eot = pt.stack([pt.cat([eot_startpos[i], output_eot[i]]) for i in range(eot_startpos.shape[0])])
  # Here we use output mask so we need to append the startpos to the output_eot before multiplied with mask(already included the startpos)
  output_eot *= mask
  eot_gt *= mask

  # Log the precision, recall, confusion_matrix and using wandb
  eot_gt_log = eot_gt.clone().cpu().detach().numpy()
  output_eot_log = output_eot.clone().cpu().detach().numpy()
  eot_metrics_log(eot_gt=eot_gt_log, output_eot=output_eot_log, lengths=lengths.cpu().detach().numpy(), flag=flag)

  # Implement from scratch
  # Flatten and concat all trajectory together
  eot_gt = pt.cat(([eot_gt[i][:lengths[i]+1] for i in range(eot_startpos.shape[0])]))
  output_eot = pt.sigmoid(pt.cat(([output_eot[i][:lengths[i]+1] for i in range(eot_startpos.shape[0])])))
  # Class weight for imbalance class problem
  pos_weight = pt.sum(eot_gt == 0)/pt.sum(eot_gt==1)
  neg_weight = 1
  # Prevent of pt.log(-value)
  eps = 1e-10
  # Calculate the BCE loss
  eot_loss = pt.mean(-((pos_weight * eot_gt * pt.log(output_eot + eps)) + (neg_weight * (1-eot_gt)*pt.log(1-output_eot + eps))))
  return eot_loss * 100

def add_noise(input_trajectory, startpos, lengths):
  factor = np.random.uniform(low=0.6, high=0.95)
  if args.noise_sd is None:
    noise_sd = np.random.uniform(low=0.3, high=1)
  else:
    noise_sd = args.noise_sd
  input_trajectory = pt.cat((startpos[..., :-1], input_trajectory), dim=1)
  input_trajectory = pt.cumsum(input_trajectory, dim=1)
  # plt.plot(np.diff(input_trajectory[0][:lengths[0]+1, 0].cpu().numpy()))
  # plt.plot(np.diff(input_trajectory[0][:lengths[0]+1, 1].cpu().numpy()))
  # plt.plot(np.diff(input_trajectory[0][:lengths[0]+1, -1].cpu().numpy()))
  noise_uv = pt.normal(mean=0.0, std=noise_sd, size=input_trajectory[..., :].shape).to(device)
  masking_noise = pt.nn.init.uniform_(pt.empty(input_trajectory[..., :].shape)).to(device) > np.random.rand(1)[0]
  n_noise = int(args.batch_size * factor)
  noise_idx = np.random.choice(a=args.batch_size, size=(n_noise,), replace=False)
  input_trajectory[noise_idx, :, :] += noise_uv[noise_idx, :, :] * masking_noise[noise_idx, :, :]
  # plt.plot(np.diff(input_trajectory[0][:lengths[0]+1, 0].cpu().numpy()))
  # plt.plot(np.diff(input_trajectory[0][:lengths[0]+1, 1].cpu().numpy()))
  # plt.plot(np.diff(input_trajectory[0][:lengths[0]+1, -1].cpu().numpy()))
  # plt.show()
  # exit()
  input_trajectory = pt.tensor(np.diff(input_trajectory.cpu().numpy(), axis=1)).to(device)
  return input_trajectory

def train(output_trajectory_train, output_trajectory_train_mask, output_trajectory_train_lengths, output_trajectory_train_startpos, output_trajectory_train_uv, input_trajectory_train, input_trajectory_train_mask, input_trajectory_train_lengths, input_trajectory_train_startpos, model, output_trajectory_val, output_trajectory_val_mask, output_trajectory_val_lengths, output_trajectory_val_startpos, output_trajectory_val_uv, input_trajectory_val, input_trajectory_val_mask, input_trajectory_val_lengths, input_trajectory_val_startpos, hidden, cell_state, optimizer, epoch, n_epochs, vis_signal, visualize_trajectory_flag=True, visualization_path='./visualize_html/'):
  # Training RNN/LSTM model on a minibatch
  # Run over each example batch
  # Initial hidden layer for the first RNN Cell
  # Train a model
  hidden = model.initHidden(batch_size=args.batch_size)
  cell_state = model.initCellState(batch_size=args.batch_size)
  # Training mode
  model.train()
  optimizer.zero_grad() # Clear existing gradients from previous epoch
  # Add noise on the fly
  input_trajectory_train_gt = input_trajectory_train.clone()
  input_trajectory_val_gt = input_trajectory_val.clone()
  if args.noise:
    input_trajectory_train = add_noise(input_trajectory=input_trajectory_train, startpos=input_trajectory_train_startpos, lengths=input_trajectory_train_lengths)
    input_trajectory_val = add_noise(input_trajectory=input_trajectory_val, startpos=input_trajectory_val_startpos, lengths=input_trajectory_val_lengths)
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
  train_eot_loss = EndOfTrajectoryLoss(output_eot=output_train_eot.clone(), eot_gt=output_trajectory_train_uv[..., -1], mask=output_trajectory_train_mask[..., -1], lengths=output_trajectory_train_lengths, eot_startpos=input_trajectory_train_startpos[..., -1], flag='Train')
  val_eot_loss = EndOfTrajectoryLoss(output_eot=output_val_eot.clone(), eot_gt=output_trajectory_val_uv[..., -1], mask=output_trajectory_val_mask[..., -1], lengths=output_trajectory_val_lengths, eot_startpos=input_trajectory_val_startpos[..., -1], flag='Validation')

  train_loss = train_eot_loss
  val_loss = val_eot_loss

  train_loss.backward() # Perform a backpropagation and calculates gradients
  pt.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip)
  optimizer.step() # Updates the weights accordingly to the gradients

  print('Train Loss : {:.3f}'.format(train_loss.item()), end=', ')
  print('Val Loss : {:.3f}'.format(val_loss.item()))
  wandb.log({'Train Loss':train_loss.item(), 'Validation Loss':val_loss.item()})

  if visualize_trajectory_flag == True and vis_signal == True:
    make_visualize(output_train_eot=output_train_eot, output_trajectory_train_startpos=output_trajectory_train_startpos, input_trajectory_train_lengths=input_trajectory_train_lengths, output_trajectory_train_maks=output_trajectory_train_mask, output_val_eot=output_val_eot, output_trajectory_val_startpos=output_trajectory_val_startpos, input_trajectory_val_lengths=input_trajectory_val_lengths, output_trajectory_val_mask=output_trajectory_val_mask, eot_gt_train=output_trajectory_train_uv[..., -1], eot_gt_val=output_trajectory_val_uv[..., -1], visualization_path=visualization_path, input_trajectory_train=input_trajectory_train, input_trajectory_val=input_trajectory_val)

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
    model = GRU(input_size=input_size, output_size=output_size)
  elif model_arch=='bigru':
    model = BiGRU(input_size=input_size, output_size=output_size)
  elif model_arch=='bigru_residual_list':
    model = BiGRUResidualList(input_size=input_size, output_size=output_size)
  elif model_arch=='bigru_residual_add':
    model = BiGRUResidualAdd(input_size=input_size, output_size=output_size)
  elif model_arch=='lstm':
    model = LSTM(input_size=input_size, output_size=output_size)
  elif model_arch=='bilstm':
    model = BiLSTM(input_size=input_size, output_size=output_size)
  else :
    print("Please input correct model architecture : gru, bigru, lstm, bilstm")
    exit()

  return model

def load_checkpoint(model, optimizer, lr_scheduler):
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
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch']
    min_val_loss = checkpoint['min_val_loss']
    return model, optimizer, start_epoch, lr_scheduler, min_val_loss

  else:
    print("[#] Checkpoint not found...")
    exit()

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
  parser.add_argument('--save_checkpoint', dest='save_checkpoint', type=str, help='Path to save a model checkpoint', required=True)
  parser.add_argument('--load_checkpoint', dest='load_checkpoint', type=str, help='Path to load a trained model checkpoint', default=None)
  parser.add_argument('--visualization_path', dest='visualization_path', type=str, help='Path to visualization directory', default='./visualize_html/')
  parser.add_argument('--wandb_name', dest='wandb_name', type=str, help='WanDB session name', default=None)
  parser.add_argument('--wandb_tags', dest='wandb_tags', type=str, help='WanDB tags name', default=None)
  parser.add_argument('--cuda_device_num', dest='cuda_device_num', type=int, help='Provide cuda device number', default=0)
  parser.add_argument('--wandb_notes', dest='wandb_notes', type=str, help='WanDB notes', default="")
  parser.add_argument('--model_arch', dest='model_arch', type=str, help='Input the model architecture(lstm, bilstm, gru, bigru)', required=True)
  parser.add_argument('--clip', dest='clip', type=int, help='Clipping gradients value', required=True)
  parser.add_argument('--noise', dest='noise', help='Noise on the fly', action='store_true')
  parser.add_argument('--no_noise', dest='noise', help='Noise on the fly', action='store_false')
  parser.add_argument('--noise_sd', dest='noise_sd', help='Std. of noise', type=float, default=None)
  args = parser.parse_args()

  # Init wandb
  wandb.init(project="ball-trajectory-estimation", name=args.wandb_name, tags=args.wandb_tags, notes=args.wandb_notes)

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
  model = get_model(input_size=n_input, output_size=n_output, model_arch=args.model_arch)
  model = model.to(device)

  # Define optimizer, learning rate, decay and scheduler parameters
  learning_rate = 0.005
  optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)
  decay_rate = 0.9
  lr_scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)
  start_epoch = 1

  # Load the checkpoint if it's available.
  if args.load_checkpoint is None:
    # Create a model
    print('===>No model checkpoint')
    print('[#] Define the Learning rate, Optimizer, Decay rate and Scheduler...')
  else:
    print('===>Load checkpoint with Optimizer state, Decay and Scheduler state')
    print('[#] Loading ... {}'.format(args.load_checkpoint))
    model, optimizer, start_epoch, lr_scheduler, min_val_loss = load_checkpoint(model, optimizer, lr_scheduler)

  print('[#]Model Architecture')
  print(model)

  # Log metrics with wandb
  wandb.watch(model)

  # Initialize the hidden and cell_state
  hidden = model.initHidden(batch_size=args.batch_size)
  cell_state = model.initCellState(batch_size=args.batch_size)

  # Training settings
  n_epochs = 10000
  decay_cycle = 100
  for epoch in range(start_epoch, n_epochs+1):
    accumulate_val_loss = []
    accumulate_train_loss = []
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
    output_trajectory_val_uv = batch_val['output'][4].to(device)

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
      output_trajectory_train_uv = batch_train['output'][4].to(device)

      # Visualize signal to make a plot and save to wandb
      # vis_signal = True if batch_idx+1 == len(trajectory_train_dataloader) else False
      vis_signal = True if epoch % 10 == 0 else False
      # Call function to train
      train_loss, val_loss, hidden, cell_state, model = train(output_trajectory_train=output_trajectory_train, output_trajectory_train_mask=output_trajectory_train_mask,
                                                                 output_trajectory_train_lengths=output_trajectory_train_lengths, output_trajectory_train_startpos=output_trajectory_train_startpos, output_trajectory_train_uv=output_trajectory_train_uv,
                                                                 input_trajectory_train=input_trajectory_train, input_trajectory_train_mask = input_trajectory_train_mask,
                                                                 input_trajectory_train_lengths=input_trajectory_train_lengths, input_trajectory_train_startpos=input_trajectory_train_startpos,
                                                                 output_trajectory_val=output_trajectory_val, output_trajectory_val_mask=output_trajectory_val_mask,
                                                                 output_trajectory_val_lengths=output_trajectory_val_lengths, output_trajectory_val_startpos=output_trajectory_val_startpos,
                                                                 input_trajectory_val=input_trajectory_val, input_trajectory_val_mask=input_trajectory_val_mask, output_trajectory_val_uv=output_trajectory_val_uv,
                                                                 input_trajectory_val_lengths=input_trajectory_val_lengths, input_trajectory_val_startpos=input_trajectory_val_startpos,
                                                                 model=model, hidden=hidden, cell_state=cell_state, visualize_trajectory_flag=args.visualize_trajectory_flag,
                                                                 optimizer=optimizer, epoch=epoch, n_epochs=n_epochs, vis_signal=vis_signal)

      accumulate_val_loss.append(val_loss)
      accumulate_train_loss.append(train_loss)
      vis_signal = False

    # Get the average loss for each epoch over entire dataset
    val_loss_per_epoch = np.mean(accumulate_val_loss)
    train_loss_per_epoch = np.mean(accumulate_train_loss)

    # Log the each epoch loss
    wandb.log({'Epoch Train Loss':train_loss_per_epoch, 'Epoch Validation Loss':val_loss_per_epoch})

    # Decrease learning rate every n_epochs % decay_cycle batch
    if epoch % decay_cycle == 0:
      lr_scheduler.step()
      for param_group in optimizer.param_groups:
        print("Stepping Learning rate to {}", param_group['lr'])

    # Save the model checkpoint every finished the epochs
    print('[#]Finish Epoch : {}/{}.........Train loss : {:.3f}, Val loss : {:.3f}'.format(epoch, n_epochs, train_loss_per_epoch, val_loss_per_epoch))
    if min_val_loss > val_loss_per_epoch:
      # Save model checkpoint
      save_checkpoint_best = '{}/{}_best.pth'.format(save_checkpoint, args.wandb_notes)
      print('[+++]Saving the best model checkpoint : Prev loss {:.3f} > Curr loss {:.3f}'.format(min_val_loss, val_loss_per_epoch))
      print('[+++]Saving the best model checkpoint to : ', save_checkpoint_best)
      min_val_loss = val_loss_per_epoch
      # Save to directory
      checkpoint = {'epoch':epoch+1, 'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict(), 'min_val_loss':min_val_loss}
      pt.save(checkpoint, save_checkpoint_best)
      pt.save(checkpoint, os.path.join(wandb.run.dir, 'checkpoint_best.pth'))

    else:
      print('[#]Not saving the best model checkpoint : Val loss {:.3f} not improved from {:.3f}'.format(val_loss_per_epoch, min_val_loss))


    if epoch % 20 == 0:
      # Save the lastest checkpoint for continue training every 10 epoch
      save_checkpoint_lastest = '{}/{}_lastest.pth'.format(save_checkpoint, args.wandb_notes)
      print('[#]Saving the lastest checkpoint to : ', save_checkpoint_lastest)
      checkpoint = {'epoch':epoch+1, 'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict(), 'min_val_loss':min_val_loss}
      pt.save(checkpoint, save_checkpoint_lastest)
      pt.save(checkpoint, os.path.join(wandb.run.dir, 'checkpoint_lastest.pth'))

  print("[#] Done")

