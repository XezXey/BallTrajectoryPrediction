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
from models.Finale.rnn import RNN
from models.Finale.lstm import LSTM
from models.Finale.bilstm import BiLSTM
from models.Finale.gru import GRU
from models.Finale.bigru import BiGRU
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

features = ['x', 'y', 'z', 'u', 'v', 'd', 'eot', 'og', 'rad', 'g']
x, y, z, u, v, d, eot, og, rad, g = range(len(features))

# marker_dict for contain the marker properties
marker_dict_gt = dict(color='rgba(0, 0, 255, 0.4)', size=4)
marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=4)
marker_dict_eot = dict(color='rgba(0, 255, 0, 0.4)', size=4)

def initialize_folder(path):
  if not os.path.exists(path):
      os.makedirs(path)

def get_model(model_arch, features_cols, args):
  if model_arch=='lstm_residual':
    model_flag = LSTMResidual(input_size=2, output_size=1, batch_size=args.batch_size)
    model_depth = LSTMResidual(input_size=2 + len(features_cols), output_size=1, batch_size=args.batch_size)
  elif model_arch=='bilstm_residual':
    model_flag = BiLSTMResidual(input_size=2, output_size=1, batch_size=args.batch_size)
    model_depth = BiLSTMResidual(input_size=2 + len(features_cols), output_size=1, batch_size=args.batch_size)
  elif model_arch=='gru_residual':
    model_flag = GRUResidual(input_size=2, output_size=1, batch_size=args.batch_size)
    model_depth = GRUResidual(input_size=2 + len(features_cols), output_size=1, batch_size=args.batch_size)
  elif model_arch=='bigru_residual':
    model_flag = BiGRUResidual(input_size=2, output_size=1, batch_size=args.batch_size)
    model_depth = BiGRUResidual(input_size=2 + len(features_cols), output_size=1, batch_size=args.batch_size)
  else :
    print("Please input correct model architecture : gru, bigru, lstm, bilstm")
    exit()

  model_cfg = {'flag':{'input_size':model_flag.input_size, 'output_size':model_flag.output_size, 'hidden_dim':model_flag.hidden_dim, 'n_layers':model_flag.n_layers, 'n_stack':model_flag.n_stack, 'recurrent_stacked':model_flag.recurrent_stacked, 'fc_size':model_flag.fc_size},
               'depth':{'input_size':model_depth.input_size, 'output_size':model_depth.output_size, 'hidden_dim':model_depth.hidden_dim, 'n_layers':model_depth.n_layers, 'n_stack':model_depth.n_stack, 'recurrent_stacked':model_depth.recurrent_stacked, 'fc_size':model_depth.fc_size}}

  return model_flag, model_depth, model_cfg

def visualize_layout_update(fig=None, n_vis=3):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  for i in range(n_vis*2):
    fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-27, 33],), yaxis = dict(nticks=5, range=[-2, 12],), zaxis = dict(nticks=10, range=[-31, 19],), aspectmode='manual', aspectratio=dict(x=4, y=2, z=3))
  return fig

def make_visualize(input_train_dict, gt_train_dict, input_val_dict, gt_val_dict, pred_train_dict, pred_val_dict, visualization_path):
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
  fig_displacement = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  visualize_displacement(uv=pred_train_dict['input'][..., [0, 1]], depth=pred_train_dict['depth'], pred_eot=pred_train_dict['flag'], gt_eot=gt_train_dict['d_with_f'][..., [1]], lengths=input_train_dict['lengths'], n_vis=n_vis, vis_idx=train_vis_idx, fig=fig_displacement, flag='Train')
  visualize_displacement(uv=pred_val_dict['input'][..., [0, 1]], depth=pred_val_dict['depth'], pred_eot=pred_val_dict['flag'], lengths=input_val_dict['lengths'], gt_eot=gt_val_dict['d_with_f'][..., [1]], n_vis=n_vis, vis_idx=val_vis_idx, fig=fig_displacement, flag='Validation')

  ####################################
  ############### EOT ################
  ####################################
  fig_eot = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  visualize_eot(pred=pred_train_dict['flag'], gt=gt_train_dict['d_with_f'][..., [1]], startpos=gt_train_dict['startpos'][..., [1]], lengths=gt_train_dict['lengths'], mask=gt_train_dict['mask'][..., [1]], fig=fig_eot, flag='Train', n_vis=n_vis, vis_idx=train_vis_idx)
  visualize_eot(pred=pred_val_dict['flag'], gt=gt_val_dict['d_with_f'][..., [1]], startpos=gt_val_dict['startpos'][..., [1]], lengths=gt_val_dict['lengths'], mask=gt_val_dict['mask'][..., [1]], fig=fig_eot, flag='Validation', n_vis=n_vis, vis_idx=val_vis_idx)

  plotly.offline.plot(fig_traj, filename='./{}/trajectory_visualization_depth_pitch_scaled.html'.format(visualization_path), auto_open=True)
  plotly.offline.plot(fig_eot, filename='./{}/trajectory_visualization_eot.html'.format(visualization_path), auto_open=True)
  plotly.offline.plot(fig_displacement, filename='./{}/trajectory_visualization_displacement.html'.format(visualization_path), auto_open=True)
  wandb.log({"PITCH SCALED : Trajectory Visualization(Col1=Train, Col2=Val)":wandb.Html(open('./{}/trajectory_visualization_depth_pitch_scaled.html'.format(visualization_path)))})
  wandb.log({"DISPLACEMENT VISUALIZATION":fig_displacement})
  wandb.log({"End Of Trajectory flag Prediction : (Col1=Train, Col2=Val)":fig_eot})

def visualize_displacement(uv, depth, pred_eot, gt_eot, lengths, vis_idx, fig=None, flag='train', n_vis=5):
  uv = uv.cpu().detach().numpy()
  pred_eot = pred_eot.cpu().detach().numpy()
  gt_eot = gt_eot.cpu().detach().numpy()
  depth = depth.cpu().detach().numpy()
  lengths = lengths.cpu().detach().numpy()
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=depth[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-Displacement of DEPTH'.format(flag, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=uv[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Displacement of U'.format(flag, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=uv[i][:lengths[i], 1], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Displacement of V'.format(flag, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=pred_eot[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_eot, name='{}-traj#{}-EOT(Pred)'.format(flag, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=gt_eot[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_eot, name='{}-traj#{}-EOT(GT)'.format(flag, i)), row=idx+1, col=col)

def visualize_trajectory(pred, gt, lengths, mask, vis_idx, fig=None, flag='Train', n_vis=5):
  # detach() for visualization
  pred = pred.cpu().detach().numpy()
  gt = gt.cpu().detach().numpy()
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter3d(x=pred[i][:lengths[i], 0], y=pred[i][:lengths[i], 1], z=pred[i][:lengths[i], 2], mode='markers', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}".format(flag, i, loss.TrajectoryLoss(pt.tensor(pred[i]).to(device), pt.tensor(gt[i]).to(device), mask=mask[i]))), row=idx+1, col=col)
    fig.add_trace(go.Scatter3d(x=gt[i][:lengths[i], 0], y=gt[i][:lengths[i], 1], z=gt[i][:lengths[i], 2], mode='markers', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+1, col=col)

def visualize_eot(pred, gt, startpos, lengths, mask, vis_idx, fig=None, flag='Train', n_vis=5):
  # pred : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  pred = pt.stack([pt.cat([startpos[i], pred[i]]) for i in range(startpos.shape[0])])
  gt = pt.stack([pt.cat([startpos[i], gt[i]]) for i in range(startpos.shape[0])])
  # Here we use output mask so we need to append the startpos to the pred before multiplied with mask(already included the startpos)
  pred *= mask
  gt *= mask
  # Weight of positive/negative classes for imbalanced class
  pos_weight = pt.sum(gt == 0)/pt.sum(gt==1)
  neg_weight = 1
  eps = 1e-10
  # Calculate the EOT loss for each trajectory
  eot_loss = pt.mean(-((pos_weight * gt * pt.log(pred+eps)) + (neg_weight * (1-gt)*pt.log(1-pred+eps))), dim=1).cpu().detach().numpy()

  # detach() for visualization
  pred = pred.cpu().detach().numpy()
  gt = gt.cpu().detach().numpy()
  lengths = lengths.cpu().detach().numpy()
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]).reshape(-1,), y=pred[i][:lengths[i], :].reshape(-1,), mode='markers+lines', marker=marker_dict_pred, name="{}-EOT Predicted [{}], EOTLoss = {:.3f}".format(flag, i, eot_loss[i][0])), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]).reshape(-1,), y=gt[i][:lengths[i], :].reshape(-1,), mode='markers+lines', marker=marker_dict_gt, name="{}-Ground Truth EOT [{}]".format(flag, i)), row=idx+1, col=col)

def get_selected_cols(args):
  # Flag/Extra features columns
  features_cols = []
  if 'eot' in args.selected_features:
    features_cols.append(eot)
  if 'og' in args.selected_features:
    features_cols.append(og)
  if 'rad' in args.selected_features:
    features_cols.append(rad)

  input_col = [u, v] + features_cols
  input_startpos_col = [u, v, d] + features_cols
  output_col = [d] + features_cols
  output_startpos_col = [x, y, z] + features_cols
  output_xyz_col = [x, y, z] + features_cols
  print('='*46 + "Features" + '='*46)
  print("Available features : ", ['{}-{}'.format(features[idx], idx) for idx in range(len(features))])
  print("Selected features : ", features_cols)
  print("1. input_col = ", input_col)
  print("2. input_startpos_col = ", input_startpos_col)
  print("3. output_col = ", output_col)
  print("4. output_startpos_col = ", output_startpos_col)
  print("5. output_xyz_col = ", output_xyz_col)
  print('='*100)
  return input_col, input_startpos_col, output_col, output_startpos_col, output_xyz_col, features_cols
