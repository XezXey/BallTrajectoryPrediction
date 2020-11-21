import os
import sys
sys.path.append(os.path.realpath('../'))
import numpy as np
import torch as pt
import json
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from mpl_toolkits import mplot3d
import wandb
# Models
# No Stack
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
# Trainable Initial State
from models.Finale.residual_init_trainable.bilstm_residual_trainable_init import BiLSTMResidualTrainableInit
from models.Finale.init_trainable.bilstm_trainable_init import BiLSTMTrainableInit
# Encoder
from models.Finale.encoder.encoder import Encoder
# Loss
import main.Finale.loss as loss

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
else:
  device = pt.device('cpu')

args=None
features = ['x', 'y', 'z', 'u', 'v', 'd', 'eot', 'og', 'rad', 'f_sin', 'f_cos', 'g']
x, y, z, u, v, d, eot, og, rad, f_sin, f_cos, g = range(len(features))

def share_args(a):
  global args
  args = a

# marker_dict for contain the marker properties
marker_dict_gt = dict(color='rgba(0, 0, 255, 0.4)', size=4)
marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=4)
marker_dict_eot = dict(color='rgba(0, 255, 0, 0.4)', size=4)

def initialize_folder(path):
  if not os.path.exists(path):
      os.makedirs(path)

def get_model_depth(model_arch, features_cols, args):

  #############################################
  ############ Prediction Selection ###########
  #############################################
  if args.bi_pred_avg or args.bi_pred_ramp:
    # Predict depth in 2 direction
    output_size = 2
  elif args.bi_pred_weight:
    # Predict depth in 2 direction
    output_size = 3
  else:
    # Predict only bw or fw depth direction
    output_size = 1

  refinement_outsize = 3
  #############################################
  ############## Model Selection ##############
  #############################################

  # Specified the size of latent 
  addition_input_size = 0
  if 'latent' in args.pipeline:
    # Create Encoder Network
    if args.latent_insize is None:
      print("[#] Please specify the size of input latent")
      exit()
    model_latent = Encoder(input_size=args.latent_insize, output_size=args.latent_outsize, batch_size=args.batch_size, model='latent')
    if 'eot' in args.pipeline:
      addition_input_size  += args.latent_outsize + 1
    else:
      addition_input_size  += args.latent_outsize
  else:
    addition_input_size = len(features_cols)

  if 'eot' in args.pipeline:
    refinement_addition_insize = addition_input_size - 1
    if 'refinement' in args.pipeline:
      addition_input_size = 1
  else:
    refinement_addition_insize = addition_input_size



  if model_arch=='lstm_residual':
    model_flag = LSTMResidual(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_depth = LSTMResidual(input_size=2 + addition_input_size, output_size=output_size, batch_size=args.batch_size, model='depth')
  elif model_arch=='bilstm_residual':
    model_flag = BiLSTMResidual(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_depth = BiLSTMResidual(input_size=2 + addition_input_size, output_size=output_size, batch_size=args.batch_size, model='depth')
  elif model_arch=='bilstm_residual_trainable_init':
    model_flag = BiLSTMResidualTrainableInit(input_size=2, output_size=1, batch_size=args.batch_size, trainable_init=args.trainable_init, bidirectional=args.bidirectional, model='flag')
    model_depth = BiLSTMResidualTrainableInit(input_size=2 + addition_input_size, output_size=output_size, batch_size=args.batch_size, trainable_init=args.trainable_init, bidirectional=args.bidirectional, model='depth')
    model_refinement = BiLSTMResidualTrainableInit(input_size=3 + refinement_addition_insize, output_size=refinement_outsize, batch_size=args.batch_size, trainable_init=args.trainable_init, bidirectional=args.bidirectional, model='refinement')
  elif model_arch=='bilstm_trainable_init':
    model_flag = BiLSTMTrainableInit(input_size=2, output_size=1, batch_size=args.batch_size, trainable_init=args.trainable_init, bidirectional=args.bidirectional, model='flag')
    model_depth = BiLSTMTrainableInit(input_size=2 + addition_input_size, output_size=output_size, batch_size=args.batch_size, trainable_init=args.trainable_init, bidirectional=args.bidirectional, model='depth')
    model_refinement = BiLSTMTrainableInit(input_size=3 + refinement_addition_insize, output_size=refinement_outsize, batch_size=args.batch_size, trainable_init=args.trainable_init, bidirectional=args.bidirectional, model='refinement')
  elif model_arch=='gru_residual':
    model_flag = GRUResidual(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_depth = GRUResidual(input_size=2 + addition_input_size, output_size=output_size, batch_size=args.batch_size, model='depth')
  elif model_arch=='bigru_residual':
    model_flag = BiGRUResidual(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_depth = BiGRUResidual(input_size=2 + addition_input_size, output_size=output_size, batch_size=args.batch_size, model='depth')
  elif model_arch=='bigru':
    model_flag = BiGRU(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_depth = BiGRU(input_size=2 + addition_input_size, output_size=output_size, batch_size=args.batch_size, model='depth')
  elif model_arch=='bilstm':
    model_flag = BiLSTM(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_depth = BiLSTM(input_size=2 + addition_input_size, output_size=output_size, batch_size=args.batch_size, model='depth')
  else :
    print("Please input correct model architecture : gru, bigru, lstm, bilstm")
    exit()

  #############################################
  ############# Pipeline Selection ############
  #############################################

  model_cfg = {}
  if 'eot' in args.pipeline:
    model_cfg['model_flag'] = {'input_size':model_flag.input_size, 'output_size':model_flag.output_size, 'hidden_dim':model_flag.hidden_dim, 'n_layers':model_flag.n_layers, 'n_stack':model_flag.n_stack, 'recurrent_stacked':model_flag.recurrent_stacked, 'fc_size':model_flag.fc_size}

  if 'depth' in args.pipeline:
    model_cfg['model_depth'] = {'input_size':model_depth.input_size, 'output_size':model_depth.output_size, 'hidden_dim':model_depth.hidden_dim, 'n_layers':model_depth.n_layers, 'n_stack':model_depth.n_stack, 'recurrent_stacked':model_depth.recurrent_stacked, 'fc_size':model_depth.fc_size}

  if 'latent' in args.pipeline:
    model_cfg['model_latent'] = {'input_size':model_latent.input_size, 'output_size':model_latent.output_size,'fc_size':model_latent.fc_size}

  if 'refinement' in args.pipeline:
    model_cfg['model_refinement'] = {'input_size':model_refinement.input_size, 'output_size':model_refinement.output_size, 'hidden_dim':model_refinement.hidden_dim, 'n_layers':model_refinement.n_layers, 'n_stack':model_refinement.n_stack, 'recurrent_stacked':model_refinement.recurrent_stacked, 'fc_size':model_refinement.fc_size}

  model_dict = {}
  for model in args.pipeline:
    if model == 'eot':
      module_name = 'flag'
    else:
      module_name = model
    model_dict['model_{}'.format(module_name)] = eval('model_{}'.format(module_name))
  return model_dict, model_cfg

def get_model_xyz(model_arch, features_cols, args):
  if model_arch=='lstm_residual':
    model_flag = LSTMResidual(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_xyz = LSTMResidual(input_size=2 + len(features_cols), output_size=3, batch_size=args.batch_size, model='xyz')
  elif model_arch=='bilstm_residual':
    model_flag = BiLSTMResidual(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_xyz = BiLSTMResidual(input_size=2 + len(features_cols), output_size=3, batch_size=args.batch_size, model='xyz')
  elif model_arch=='gru_residual':
    model_flag = GRUResidual(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_xyz = GRUResidual(input_size=2 + len(features_cols), output_size=3, batch_size=args.batch_size, model='xyz')
  elif model_arch=='bigru_residual':
    model_flag = BiGRUResidual(input_size=2, output_size=1, batch_size=args.batch_size, model='flag')
    model_xyz = BiGRUResidual(input_size=2 + len(features_cols), output_size=3, batch_size=args.batch_size, model='xyz')
  else :
    print("Please input correct model architecture : gru, bigru, lstm, bilstm")
    exit()

  model_cfg = {'flag':{'input_size':model_flag.input_size, 'output_size':model_flag.output_size, 'hidden_dim':model_flag.hidden_dim, 'n_layers':model_flag.n_layers, 'n_stack':model_flag.n_stack, 'recurrent_stacked':model_flag.recurrent_stacked, 'fc_size':model_flag.fc_size},
               'xyz':{'input_size':model_xyz.input_size, 'output_size':model_xyz.output_size, 'hidden_dim':model_xyz.hidden_dim, 'n_layers':model_xyz.n_layers, 'n_stack':model_xyz.n_stack, 'recurrent_stacked':model_xyz.recurrent_stacked, 'fc_size':model_xyz.fc_size}}

  return model_flag, model_xyz, model_cfg

def visualize_layout_update(fig=None, n_vis=3):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  for i in range(n_vis*2):
    fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-10, 10],), yaxis = dict(nticks=5, range=[-2, 3],), zaxis = dict(nticks=10, range=[-10, 10],), aspectmode='manual', aspectratio=dict(x=4, y=2, z=3))
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
  # fig_traj = visualize_layout_update(fig=fig_traj, n_vis=n_vis)

  ####################################
  ########### Displacement ###########
  ####################################
  if pred == 'xyz':
    eot_gt_col = 3
  elif pred == 'depth':
    eot_gt_col = 1

  if 'eot' in args.pipeline:
    gt_eot_train = gt_train_dict['o_with_f'][..., [eot_gt_col]]
    pred_eot_train = pred_train_dict['flag']
    gt_eot_val = gt_val_dict['o_with_f'][..., [eot_gt_col]]
    pred_eot_val = pred_val_dict['flag']
  else:
    gt_eot_train  = None
    pred_eot_train  = None
    gt_eot_val  = None
    pred_eot_val  = None

  fig_displacement = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  visualize_displacement(uv=pred_train_dict['input'][..., [0, 1]], depth=pred_train_dict[pred], pred_eot=pred_eot_train, gt_eot=gt_eot_train, lengths=input_train_dict['lengths'], n_vis=n_vis, vis_idx=train_vis_idx, fig=fig_displacement, flag='Train', pred=pred)
  visualize_displacement(uv=pred_val_dict['input'][..., [0, 1]], depth=pred_val_dict[pred], pred_eot=pred_eot_val, lengths=input_val_dict['lengths'], gt_eot=gt_eot_val, n_vis=n_vis, vis_idx=val_vis_idx, fig=fig_displacement, flag='Validation', pred=pred)

  ####################################
  ############### EOT ################
  ####################################

  if 'eot' in args.pipeline:
    fig_eot = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
    visualize_eot(pred=pred_train_dict['flag'], gt=gt_train_dict['o_with_f'][..., [eot_gt_col]], startpos=gt_train_dict['startpos'][..., [3]], lengths=gt_train_dict['lengths'], mask=gt_train_dict['mask'][..., [3]], fig=fig_eot, flag='Train', n_vis=n_vis, vis_idx=train_vis_idx)
    visualize_eot(pred=pred_val_dict['flag'], gt=gt_val_dict['o_with_f'][..., [eot_gt_col]], startpos=gt_val_dict['startpos'][..., [3]], lengths=gt_val_dict['lengths'], mask=gt_val_dict['mask'][..., [3]], fig=fig_eot, flag='Validation', n_vis=n_vis, vis_idx=val_vis_idx)
    plotly.offline.plot(fig_eot, filename='./{}/trajectory_visualization_eot.html'.format(visualization_path), auto_open=True)
    wandb.log({"End Of Trajectory flag Prediction : (Col1=Train, Col2=Val)":fig_eot})

  plotly.offline.plot(fig_traj, filename='./{}/trajectory_visualization_depth_pitch_scaled.html'.format(visualization_path), auto_open=True)
  plotly.offline.plot(fig_displacement, filename='./{}/trajectory_visualization_displacement.html'.format(visualization_path), auto_open=True)
  wandb.log({"PITCH SCALED : Trajectory Visualization(Col1=Train, Col2=Val)":wandb.Html(open('./{}/trajectory_visualization_depth_pitch_scaled.html'.format(visualization_path)))})
  wandb.log({"DISPLACEMENT VISUALIZATION":fig_displacement})

def visualize_displacement(uv, depth, pred_eot, gt_eot, lengths, vis_idx, pred, fig=None, flag='train', n_vis=5):
  uv = uv.cpu().detach().numpy()
  depth = depth.cpu().detach().numpy()
  lengths = lengths.cpu().detach().numpy()
  if pred_eot is not None:
    pred_eot = pred_eot.cpu().detach().numpy()
  if gt_eot is not None:
    gt_eot = gt_eot.cpu().detach().numpy()
  # Change the columns for each set
  if flag == 'Train': col = 1
  elif flag == 'Validation': col=2
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    if pred=='depth':
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=depth[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-Displacement of DEPTH'.format(flag, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=uv[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Displacement of U'.format(flag, i)), row=idx+1, col=col)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=uv[i][:lengths[i], 1], mode='markers+lines', marker=marker_dict_gt, name='{}-traj#{}-Displacement of V'.format(flag, i)), row=idx+1, col=col)
    if pred_eot is not None:
      fig.add_trace(go.Scatter(x=np.arange(lengths[i]), y=pred_eot[i][:lengths[i], 0], mode='markers+lines', marker=marker_dict_pred, name='{}-traj#{}-EOT(Pred)'.format(flag, i)), row=idx+1, col=col)
    if gt_eot is not None:
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
    fig.add_trace(go.Scatter3d(x=pred[i][:lengths[i], 0], y=pred[i][:lengths[i], 1], z=pred[i][:lengths[i], 2], mode='markers+lines', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}".format(flag, i, loss.TrajectoryLoss(pt.tensor(pred[i]).to(device), pt.tensor(gt[i]).to(device), mask=mask[i]))), row=idx+1, col=col)
    fig.add_trace(go.Scatter3d(x=gt[i][:lengths[i], 0], y=gt[i][:lengths[i], 1], z=gt[i][:lengths[i], 2], mode='markers+lines', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+1, col=col)

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

def reverse_masked_seq(seq, lengths):
  for i in range(seq.shape[0]):
    seq[i][:lengths[i]] = pt.flip(seq[i][:lengths[i], [0]], dims=[0])
  return seq

def construct_bipred_weight(weight, lengths):
  weight_scaler = pt.nn.Sigmoid()
  weight = weight_scaler(weight)
  fw_weight = pt.cat((pt.ones(weight.shape[0], 1, 1).cuda(), weight), dim=1)
  bw_weight = pt.cat((pt.zeros(weight.shape[0], 1, 1).cuda(), 1-weight), dim=1)
  for i in range(weight.shape[0]):
    # Forward prediction weight
    fw_weight[i][lengths[i]] = 0.
    # Bachward prediction weight
    bw_weight[i][lengths[i]] = 1.
    # print(pt.cat((fw_weight, bw_weight), dim=2)[i][:lengths[i]+20])
    # exit()
  # print(weight.shape, lengths.shape)

  return pt.cat((fw_weight, bw_weight), dim=2)

def construct_bipred_ramp(weight_template, lengths):
  fw_weight = pt.zeros(weight_template.shape[0], weight_template.shape[1]+1, weight_template.shape[2]).cuda()
  bw_weight = pt.zeros(weight_template.shape[0], weight_template.shape[1]+1, weight_template.shape[2]).cuda()
  # print(fw_weight.shape, bw_weight.shape)
  for i in range(weight_template.shape[0]):
    # Forward prediction weight
    fw_weight[i][:lengths[i]+1] = 1 - pt.linspace(start=0, end=1, steps=lengths[i]+1).view(-1, 1).to(device)
    # Backward prediction weight
    bw_weight[i][:lengths[i]+1] = pt.linspace(start=0, end=1, steps=lengths[i]+1).view(-1, 1).to(device)

  return pt.cat((fw_weight, bw_weight), dim=2)

def save_reconstructed(eval_metrics, trajectory):
  # Take the evaluation metrics and reconstructed trajectory to create the save file for ranking visualization
  lengths = []
  trajectory_all = []
  for i in range(len(trajectory)):
    # Iterate over each batch
    gt_xyz = trajectory[i][0]
    pred_xyz = trajectory[i][1]
    uv = trajectory[i][2]
    d = trajectory[i][3]
    seq_len = trajectory[i][4]
    for j in range(seq_len.shape[0]):
      # Iterate over each trajectory
      # print(gt_xyz[j].shape, pred_xyz[j].shape, uv[j].shape, d[j].shape)
      each_trajectory = np.concatenate((gt_xyz[j][:seq_len[j]], pred_xyz[j][:seq_len[j]], uv[j][:seq_len[j]], d[j][:seq_len[j]].reshape(-1, 1)), axis=1)
      lengths.append(seq_len)
      trajectory_all.append(each_trajectory)

  # Save to file
  save_file_suffix = args.load_checkpoint.split('/')[-2]
  save_path = '{}/{}'.format(args.savetofile, save_file_suffix)
  initialize_folder(save_path)
  np.save(file='{}/{}_trajectory'.format(save_path, save_file_suffix), arr=np.array(trajectory_all))
  np.save(file='{}/{}_metrices'.format(save_path, save_file_suffix), arr=eval_metrics)
  print("[#] Saving reconstruction to /{}/{}".format(args.savetofile, save_file_suffix))

def save_visualize(fig, postfix=None):
  if postfix is None:
    postfix = 0
  save_file_suffix = args.load_checkpoint.split('/')[-2]
  save_path = '{}/{}'.format(args.savetofile, save_file_suffix)
  initialize_folder(save_path)
  plotly.offline.plot(fig, filename='./{}/interactive_optimize_{}.html'.format(save_path, postfix), auto_open=False)

def add_noise(input_trajectory, startpos, lengths):
  factor = np.random.uniform(low=0.6, high=0.95)
  if args.noise_sd is None:
    noise_sd = np.random.uniform(low=0.3, high=1)
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
  n_noise = int(input_trajectory.shape[0] * factor)
  noise_idx = np.random.choice(a=input_trajectory.shape[0], size=(n_noise,), replace=False)
  input_trajectory[noise_idx] += noise_uv[noise_idx] * masking_noise[noise_idx]
  input_trajectory = pt.tensor(np.diff(input_trajectory.cpu().numpy(), axis=1)).to(device)
  return input_trajectory

'''
def print_loss(train, val, suffix):
  train_loss_dict = train[0]
  train_loss = train[1]
  val_loss_dict = val[0]
  val_loss = val[1]
  print('\n   [##] Training...', end=' ')
  print('Train Loss : {:.3f}'.format(train_loss.item()))
  for idx, loss in enumerate(train_loss_dict.keys()):
    if idx == 0:
      print('   ======> {} : {:.3f}'.format(loss, train_loss_dict[loss]), end=', ')
    elif idx == len(train_loss_dict.keys())-1:
      print('{} : {:.3f}'.format(loss, train_loss_dict[loss]))
    else:
      print('{} : {:.3f}'.format(loss, train_loss_dict[loss]), end=', ')

  print('   [##] Validating...', end=' ')
  print('Val Loss : {:.3f}'.format(val_loss.item()))
  for idx, loss in enumerate(val_loss_dict.keys()):
    if idx == 0:
      print('   ======> {} : {:.3f}'.format(loss, val_loss_dict[loss]), end=', ')
    elif idx == len(val_loss_dict.keys())-1:
      print('{} : {:.3f}'.format(loss, val_loss_dict[loss]))
    else:
      print('{} : {:.3f}'.format(loss, val_loss_dict[loss]), end=', ')
'''

def print_loss(loss_list, name):
  loss_dict = loss_list[0]
  loss = loss_list[1]
  print('   [##] {}...'.format(name), end=' ')
  print('{} Loss : {:.3f}'.format(name, loss.item()))
  for idx, loss in enumerate(loss_dict.keys()):
    if idx == 0:
      print('   ======> {} : {:.3f}'.format(loss, loss_dict[loss]), end=', ')
    elif idx == len(loss_dict.keys())-1:
      print('{} : {:.3f}'.format(loss, loss_dict[loss]))
    else:
      print('{} : {:.3f}'.format(loss, loss_dict[loss]), end=', ')

def add_flag_noise(flag, lengths):
  flag = flag * 0
  return flag

def get_pipeline_var(pred_dict, input_dict):
  pred_flag = None
  input_flag = None
  if 'eot' in args.pipeline:
    pred_flag = pred_dict['model_flag']
    if args.env == 'unity':
      input_flag = input_dict['input'][..., [2]]
  if 'depth' in args.pipeline:
    pred_depth = pred_dict['model_depth']

  return pred_depth, pred_flag, input_flag

def get_extrinsic_representation(cam_params_dict):
  print(cam_params_dict)
