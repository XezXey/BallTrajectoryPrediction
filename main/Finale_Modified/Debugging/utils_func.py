import os
import sys
sys.path.append(os.path.realpath('../'))
import numpy as np
import torch as pt
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from mpl_toolkits import mplot3d
import wandb
# Models
# No stack
# from models.Finale.vanilla_nostack.lstm import LSTM
# from models.Finale.vanilla_nostack.bilstm import BiLSTM
# from models.Finale.vanilla_nostack.gru import GRU
# from models.Finale.vanilla_nostack.bigru import BiGRU
# Stack
from models.Finale.vanilla_stack.lstm import LSTM
from models.Finale.vanilla_stack.bilstm import BiLSTM
from models.Finale.vanilla_stack.gru import GRU
from models.Finale.vanilla_stack.bigru import BiGRU
# Residual
from models.Finale.residual.bilstm_residual import BiLSTMResidual
from models.Finale.residual.lstm_residual import LSTMResidual
from models.Finale.residual.bigru_residual import BiGRUResidual
from models.Finale.residual.gru_residual import GRUResidual
# Loss
import main.Finale.loss as loss

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
else:
  device = pt.device('cpu')

args = None
features = ['x', 'y', 'z', 'u', 'v', 'd', 'eot', 'og', 'rad', 'f_sin', 'f_cos', 'g']
x, y, z, u, v, d, eot, og, rad, f_sin, f_cos, g = range(len(features))

# marker_dict for contain the marker properties
marker_dict_gt = dict(color='rgba(0, 0, 255, 0.4)', size=4)
marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=4)
marker_dict_latent = dict(color='rgba(0, 255, 0, 0.4)', size=4)

def initialize_folder(path):
  if not os.path.exists(path):
      os.makedirs(path)

def share_args(a):
  global args
  args = a

def get_model_depth(model_arch, features_cols):
  if args.latent:
    input_size = 3
  else:
    input_size = 2

  if model_arch=='bilstm_residual':
    model_flag = BiLSTMResidual(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_depth = BiLSTMResidual(input_size=input_size, output_size=1, batch_size=args.batch_size, model='depth')
  elif model_arch=='bigru_residual':
    model_flag = BiGRUResidual(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_depth = BiGRUResidual(input_size=input_size, output_size=1, batch_size=args.batch_size, model='depth')
  elif model_arch=='bilstm':
    model_flag = BiLSTM(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_depth = BiLSTM(input_size=input_size, output_size=1, batch_size=args.batch_size, model='depth')
  elif model_arch=='bigru':
    model_flag = BiGRU(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_depth = BiGRU(input_size=input_size, output_size=1, batch_size=args.batch_size, model='depth')
  elif model_arch=='lstm':
    model_flag = LSTM(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_depth = LSTM(input_size=input_size, output_size=1, batch_size=args.batch_size, model='depth')
  elif model_arch=='gru':
    model_flag = GRU(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_depth = GRU(input_size=input_size, output_size=1, batch_size=args.batch_size, model='depth')
  else :
    print("Please input correct model architecture : gru, bigru, lstm, bilstm")
    exit()

  model_cfg = {'flag':{'input_size':model_flag.input_size, 'output_size':model_flag.output_size, 'hidden_dim':model_flag.hidden_dim, 'n_layers':model_flag.n_layers, 'n_stack':model_flag.n_stack, 'recurrent_stacked':model_flag.recurrent_stacked, 'fc_size':model_flag.fc_size},
               'depth':{'input_size':model_depth.input_size, 'output_size':model_depth.output_size, 'hidden_dim':model_depth.hidden_dim, 'n_layers':model_depth.n_layers, 'n_stack':model_depth.n_stack, 'recurrent_stacked':model_depth.recurrent_stacked, 'fc_size':model_depth.fc_size}}

  return model_flag, model_depth, model_cfg

def get_model_xyz(model_arch, features_cols):
  if args.latent:
    input_size = 3
  else:
    input_size = 2
  if model_arch=='bilstm_residual':
    model_flag = BiLSTMResidual(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_xyz = BiLSTMResidual(input_size=input_size, output_size=1, batch_size=args.batch_size, model='xyz')
  elif model_arch=='bigru_residual':
    model_flag = BiGRUResidual(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_xyz = BiGRUResidual(input_size=input_size, output_size=1, batch_size=args.batch_size, model='xyz')
  else :
    print("Please input correct model architecture : gru, bigru, lstm, bilstm")
    exit()

  model_cfg = {'flag':{'input_size':model_flag.input_size, 'output_size':model_flag.output_size, 'hidden_dim':model_flag.hidden_dim, 'n_layers':model_flag.n_layers, 'n_stack':model_flag.n_stack, 'recurrent_stacked':model_flag.recurrent_stacked, 'fc_size':model_flag.fc_size},
               'xyz':{'input_size':model_xyz.input_size, 'output_size':model_xyz.output_size, 'hidden_dim':model_xyz.hidden_dim, 'n_layers':model_xyz.n_layers, 'n_stack':model_xyz.n_stack, 'recurrent_stacked':model_xyz.recurrent_stacked, 'fc_size':model_xyz.fc_size}}

  return model_flag, model_xyz, model_cfg

def visualize_layout_update(fig=None, n_vis=3):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  for i in range(n_vis*2):
    fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-27, 33],), yaxis = dict(nticks=5, range=[-2, 12],), zaxis = dict(nticks=10, range=[-31, 19],), aspectmode='manual', aspectratio=dict(x=4, y=2, z=3))
  return fig

def make_visualize(input_train_dict, gt_train_dict, input_val_dict, gt_val_dict, pred_train_dict, pred_val_dict, visualization_path, pred):
  # Visualize by make a subplots of trajectory
  n_vis = 5
  # Random the index the be visualize
  train_vis_idx = np.random.randint(low=0, high=input_train_dict['input'].shape[0], size=(n_vis))
  val_vis_idx = np.random.randint(low=0, high=input_val_dict['input'].shape[0], size=(n_vis))

  ####################################
  ############ Trajectory ############
  ####################################
  fig_traj = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  # Append the start position and apply cummulative summation for transfer the displacement to the x, y, z coordinate. These will done by visualize_trajectory function
  # Can use mask directly because the mask obtain from full trajectory(Not remove the start pos)
  visualize_trajectory(pred=pt.mul(pred_train_dict['xyz'], gt_train_dict['mask'][..., [0, 1, 2]]), gt=gt_train_dict['xyz'][..., [0, 1, 2]], lengths=gt_train_dict['lengths'], mask=gt_train_dict['mask'][..., [0, 1, 2]], fig=fig_traj, flag='Train', n_vis=n_vis, vis_idx=train_vis_idx)
  visualize_trajectory(pred=pt.mul(pred_val_dict['xyz'], gt_val_dict['mask'][..., [0, 1, 2]]), gt=gt_val_dict['xyz'][..., [0, 1, 2]], lengths=gt_val_dict['lengths'], mask=gt_val_dict['mask'][..., [0, 1, 2]], fig=fig_traj, flag='Validation', n_vis=n_vis, vis_idx=val_vis_idx)
  fig_traj.update_layout(height=1920, width=1500, autosize=True) # Adjust the layout/axis for pitch scale
  fig_traj = visualize_layout_update(fig=fig_traj, n_vis=n_vis)

  ####################################
  ########### Displacement ###########
  ####################################
  if pred == 'xyz':
    latent_gt_col = 3
  elif pred == 'depth':
    latent_gt_col = 1

  fig_displacement = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)

  if args.latent:
    latent_in = [pred_train_dict['input'][..., 2:].cpu().detach().numpy(), pred_val_dict['input'][..., 2:].cpu().detach().numpy()]
  else:
    latent_in = [None, None]
  visualize_displacement(uv=pred_train_dict['input'][..., [0, 1]], depth=pred_train_dict[pred], latent_in=latent_in[0], latent_gt=gt_train_dict['o_with_f'][..., [latent_gt_col]], lengths=input_train_dict['lengths'], n_vis=n_vis, vis_idx=train_vis_idx, fig=fig_displacement, flag='Train', pred=pred)
  visualize_displacement(uv=pred_val_dict['input'][..., [0, 1]], depth=pred_val_dict[pred], latent_in=latent_in[1], lengths=input_val_dict['lengths'], latent_gt=gt_val_dict['o_with_f'][..., [latent_gt_col]], n_vis=n_vis, vis_idx=val_vis_idx, fig=fig_displacement, flag='Validation', pred=pred)

  ####################################
  ############# Log/Plot #############
  ####################################

  plotly.offline.plot(fig_traj, filename='./{}/trajectory_visualization_depth_pitch_scaled.html'.format(visualization_path), auto_open=True)
  plotly.offline.plot(fig_displacement, filename='./{}/trajectory_visualization_displacement.html'.format(visualization_path), auto_open=True)
  wandb.log({"PITCH SCALED : Trajectory Visualization(Col1=Train, Col2=Val)":wandb.Html(open('./{}/trajectory_visualization_depth_pitch_scaled.html'.format(visualization_path)))})
  wandb.log({"DISPLACEMENT VISUALIZATION":fig_displacement})

def visualize_displacement(uv, depth, latent_in, latent_gt, lengths, vis_idx, pred, fig=None, flag='train', n_vis=5):
  uv = uv.cpu().detach().numpy()
  latent_gt = latent_gt.cpu().detach().numpy()
  depth = depth.cpu().detach().numpy()
  lengths = lengths.cpu().detach().numpy()
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=uv[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Displacement of U'.format(flag, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=uv[i][:lengths[i], 1], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Displacement of V'.format(flag, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=latent_gt[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_latent, name='{}-traj#{}-Latent(GT)'.format(flag, i)), row=idx+1, col=col)

    if pred=='depth': # Visualize depth displacement
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=depth[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-Displacement of DEPTH'.format(flag, i)), row=idx+1, col=col)
    if args.latent: # Visualize latent space
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=latent_in[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-Latent(Input)'.format(flag, i)), row=idx+1, col=col)

def visualize_trajectory(pred, gt, lengths, mask, vis_idx, fig=None, flag='Train', n_vis=5):
  # detach() for visualization
  pred = pred.cpu().detach().numpy()
  gt = gt.cpu().detach().numpy()
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter3d(x=pred[i][:lengths[i], 0], y=pred[i][:lengths[i], 1], z=pred[i][:lengths[i], 2], mode='markers+lines', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}".format(flag, i, loss.TrajectoryLoss(pt.tensor(pred[i]).to(device), pt.tensor(gt[i]).to(device), mask=mask[i]))), row=idx+1, col=col)
    fig.add_trace(go.Scatter3d(x=gt[i][:lengths[i], 0], y=gt[i][:lengths[i], 1], z=gt[i][:lengths[i], 2], mode='markers+lines', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+1, col=col)

def get_selected_cols(args, pred):
  # Flag/Extra features columns
  features_cols = []
  if 'eot' in args.selected_features:
    features_cols.append(eot)
  if 'og' in args.selected_features:
    features_cols.append(og)
  if 'rad' in args.selected_features:
    features_cols.append(rad)
  if 'f_sin' in args.selected_features:
    features_cols.append(f_sin)
  if 'f_cos' in args.selected_features:
    features_cols.append(f_cos)
  if 'depth' in args.selected_features:
    features_cols.append(d)

  if pred=='depth' and args.env=='unity':
    input_col = [u, v] + features_cols
    input_startpos_col = [u, v, d] + features_cols
    gt_col = [d] + features_cols
    gt_startpos_col = [x, y, z] + features_cols
    gt_xyz_col = [x, y, z] + features_cols
  elif pred=='xyz' and args.env=='unity':
    input_col = [u, v] + features_cols
    input_startpos_col = [u, v, x, y, z] + features_cols
    gt_col = [x, y, z] + features_cols
    gt_startpos_col = [x, y, z] + features_cols
    gt_xyz_col = [x, y, z] + features_cols
  elif pred=='depth' and args.env=='mocap':
    input_col = [u, v]
    input_startpos_col = [u, v, d]
    gt_col = [d]
    gt_startpos_col = [x, y, z]
    gt_xyz_col = [x, y, z]
  elif pred=='xyz' and args.env=='mocap':
    input_col = [u, v]
    input_startpos_col = [u, v, x, y, z]
    gt_col = [x, y, z]
    gt_startpos_col = [x, y, z]
    gt_xyz_col = [x, y, z]

  print('='*46 + "Features" + '='*46)
  print('Prediction = {}, Environment = {}'.format(pred, args.env))
  print("Available features : ", ['{}-{}'.format(features[idx], idx) for idx in range(len(features))])
  print("Selected features : ", features_cols)
  print("1. input_col = ", input_col)
  print("2. input_startpos_col = ", input_startpos_col)
  print("3. gt_col = ", gt_col)
  print("4. gt_startpos_col = ", gt_startpos_col)
  print("5. gt_xyz_col = ", gt_xyz_col)
  print('='*100)
  return input_col, input_startpos_col, gt_col, gt_startpos_col, gt_xyz_col, features_cols