from __future__ import print_function
# Import libs
import numpy as np
np.set_printoptions(precision=4)
import torch as pt
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
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import wandb
import json
# Utils
from utils.dataloader import TrajectoryDataset
import utils.utils_func as utils_func
import utils.cummulative_depth as utils_cummulative
import utils.transformation as utils_transform
import utils.utils_inference_func as utils_inference_func
import utils.utils_model as utils_model
# Loss
import utils.loss as utils_loss
# Analysis model
from models.Finale.optimization.optimization_refinement_ErrAnalysis import OptimizationLatentAnalyze
# marker_dict for contain the marker properties
marker_dict_gt = dict(color='rgba(0, 0, 255, 0.4)', size=4)
marker_dict_noisy = dict(color='rgba(204, 102, 0, 0.4)', size=4)
marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=4)
marker_dict_axis = [dict(color='rgba(255, 0, 0, 0.7)', size=2.5), dict(color='rgba(0, 0, 255, 0.7)', size=2.5)]
marker_dict_latent = dict(color='rgba(11, 102, 35, 0.7)', size=7)
batch_ptr = 0


# Argumentparser for input
parser = argparse.ArgumentParser(description='Predict the 3D projectile')
parser.add_argument('--dataset_test_path', dest='dataset_test_path', type=str, help='Path to testing set', required=True)
parser.add_argument('--batch_size', dest='batch_size', type=int, help='Size of batch', default=50)
parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, help='Type of trajectory(Rolling, Projectile, MagnusProjectile)', default='Projectile')
parser.add_argument('--no_visualize', dest='vis_flag', help='No Visualize the trajectory', action='store_false')
parser.add_argument('--visualize', dest='vis_flag', help='Visualize the trajectory', action='store_true')
parser.add_argument('--load_checkpoint', dest='load_checkpoint', type=str, help='Path to load a tested model checkpoint', default=None)
parser.add_argument('--visualization_path', dest='visualization_path', type=str, help='Path to visualization directory', default='./visualize_html/')
parser.add_argument('--cam_params_file', dest='cam_params_file', type=str, help='Path to camera parameters file(Intrinsic/Extrinsic)')
parser.add_argument('--cuda_device_num', dest='cuda_device_num', type=int, help='Provide cuda device number', default=0)
parser.add_argument('--model_arch', dest='model_arch', type=str, help='Input the model architecture(lstm, bilstm, gru, bigru)', required=True)
parser.add_argument('--threshold', dest='threshold', type=float, help='Provide the error threshold of reconstructed trajectory', default=0.8)
parser.add_argument('--animation', dest='animation', help='Animated visualize flag', action='store_true', default=False)
parser.add_argument('--noise', dest='noise', help='Noise on the fly', action='store_true')
parser.add_argument('--flag_noise', dest='flag_noise', help='Flag noise on the fly', action='store_true', default=False)
parser.add_argument('--no_noise', dest='noise', help='Noise on the fly', action='store_false')
parser.add_argument('--noise_sd', dest='noise_sd', help='Std. of noise', type=float, default=None)
parser.add_argument('--decumulate', help='Decumulate the depth by ray casting', action='store_true', default=False)
parser.add_argument('--start_decumulate', help='Epoch to start training with decumulate of an error', type=int, default=0)
parser.add_argument('--teacherforcing_depth', help='Use a teacher forcing training scheme for depth displacement estimation', action='store_true', default=False)
parser.add_argument('--teacherforcing_mixed', help='Use a teacher forcing training scheme for depth displacement estimation on some part of training set', action='store_true', default=False)
parser.add_argument('--selected_features', dest='selected_features', help='Specify the selected features columns(eot, og, ', nargs='+', default=[])
parser.add_argument('--bi_pred_avg', help='Bidirectional prediction', action='store_true', default=False)
parser.add_argument('--bi_pred_weight', help='Bidirectional prediction with weight', action='store_true', default=False)
parser.add_argument('--bw_pred', help='Backward prediction', action='store_true', default=False)
parser.add_argument('--bi_pred_ramp', help='Bidirectional prediction with ramp weight', action='store_true', default=False)
parser.add_argument('--env', dest='env', help='Environment', type=str, default='unity')
parser.add_argument('--bidirectional', dest='bidirectional', help='Bidirectional', action='store_true')
parser.add_argument('--directional', dest='bidirectional', help='Directional', action='store_false')
parser.add_argument('--trainable_init', help='Trainable initial state', action='store_true', default=False)
parser.add_argument('--savetofile', dest='savetofile', help='Save the prediction trajectory for doing optimization', type=str, default=None)
parser.add_argument('--multiview_loss', dest='multiview_loss', help='Use multiview loss', nargs='+', default=[])
parser.add_argument('--round', dest='round', help='Rounding pixel', action='store_true', default=False)
parser.add_argument('--pipeline', dest='pipeline', help='Pipeline', nargs='+', default=[])
parser.add_argument('--n_refinement', dest='n_refinement', help='Refinement Iterations', type=int, default=1)
parser.add_argument('--fix_refinement', dest='fix_refinement', help='Fix Refinement for 1st and last points', action='store_true', default=False)
parser.add_argument('--optimize', dest='optimize', help='Flag to optimze(This will work when train with latent', action='store_true', default=False)
parser.add_argument('--latent_code', dest='latent_code', help='Optimze the latent code)', nargs='+', default=[])
parser.add_argument('--missing', dest='missing', help='Adding a missing data points while training', default=None)
parser.add_argument('--recon', dest='recon', help='Using Ideal or Noisy uv for reconstruction', default='ideal_uv')
parser.add_argument('--error_analysis', dest='error_analysis', help='Doing an error analysis plot', action='store_true', default=False)
parser.add_argument('--in_refine', dest='in_refine', help='Input for refinement network', default='xyz')
parser.add_argument('--out_refine', dest='out_refine', help='Output for refinement network', default='xyz')
parser.add_argument('--annealing', dest='annealing', help='Apply annealing', action='store_true', default=False)
parser.add_argument('--annealing_cycle', dest='annealing_cycle', type=int, help='Apply annealing every n epochs', default=5)
parser.add_argument('--annealing_gamma', dest='annealing_gamma', type=float, help='Apply annealing every n epochs', default=0.95)


args = parser.parse_args()
# Share args to every modules
utils_func.share_args(args)
utils_inference_func.share_args(args)
utils_cummulative.share_args(args)
utils_transform.share_args(args)
utils_model.share_args(args)
utils_loss.share_args(args)

# GPU initialization
if pt.cuda.is_available():
  pt.cuda.set_device(args.cuda_device_num)
  device = pt.device('cuda')
  print('[%]GPU Enabled')
else:
  device = pt.device('cpu')
  print('[%]GPU Disabled, CPU Enabled')

# Get selected features to input into a network
features = ['x', 'y', 'z', 'u', 'v', 'd', 'eot', 'og', 'rad', 'f_sin', 'f_cos', 'fx', 'fy', 'fz', 'fx_norm', 'fy_norm', 'fz_norm', 'g']
x, y, z, u, v, d, eot, og, rad, f_sin, f_cos, fx, fy, fz, fx_norm, fy_norm, fz_norm, g = range(len(features))
input_col, input_startpos_col, gt_col, gt_startpos_col, gt_xyz_col, features_cols = utils_func.get_selected_cols(args=args, pred='depth')

def add_flag_noise(flag, lengths):
  flag = flag * 0.
  # for idx in range(flag.shape[0]):
    # print(flag)
    # flag_active = pt.where(flag[idx]==1.)
    # exit()
  return flag

def predict(input_test_dict, gt_test_dict, model_dict, threshold, cam_params_dict, vis_flag=True, visualization_path='./visualize_html/'):
  # Testing RNN/LSTM model
  # Run over each example
  # Test a model

  ####################################
  ############# Testing ##############
  ####################################
  # Evaluating mode
  # utils_model.eval_mode(model_dict=model_dict)

  # Add noise on the fly
  if args.round:
    input_test_dict['input'][..., [0, 1]] = pt.round(input_test_dict['input'][..., [0, 1]])
    input_test_dict['startpos'][..., [0, 1]] = pt.round(input_test_dict['startpos'][..., [0, 1]])
  in_test = input_test_dict['input'][..., [0, 1]].clone()
  if args.noise:
    in_test = utils_func.add_noise(input_trajectory=in_test[..., [0, 1]].clone(), startpos=input_test_dict['startpos'][..., [0, 1]], lengths=input_test_dict['lengths'])

  if args.optimize and  'refinement' not in args.pipeline:
    # Optimize with Depth estimation network
    pred_xyz_test, pred_dict_test, missing_dict_test, input_uv_cumsum_test, pred_depth_cumsum_test = utils_model.optimize_depth(model_dict=model_dict, gt_dict=gt_test_dict, cam_params_dict=cam_params_dict, optimize=args.optimize, input_dict=input_test_dict)
    pred_depth_test, pred_flag_test, input_flag_test = utils_func.get_pipeline_var(pred_dict=pred_dict_test, input_dict=input_test_dict)

  else:
    pred_dict_test, in_test, missing_dict_test = utils_model.fw_pass(model_dict, input_dict=input_test_dict, cam_params_dict=cam_params_dict)

    pred_depth_test, pred_flag_test, input_flag_test = utils_func.get_pipeline_var(pred_dict=pred_dict_test, input_dict=input_test_dict)

    if args.bi_pred_weight:
      bi_pred_weight_test = pred_depth_test[..., [2]]
    else:
      bi_pred_weight_test = pt.zeros(pred_depth_test[..., [0]].shape)

    if args.missing != None:
      uv_test = utils_func.interpolate_missing(input_dict=input_test_dict, pred_dict=pred_dict_test, in_missing=in_test, missing_dict=missing_dict_test)
    elif args.recon == 'ideal_uv':
      uv_test = input_test_dict['input'][..., [0, 1]]
    elif args.recon == 'noisy_uv':
      uv_test = in_test[..., [0, 1]]

    pred_depth_cumsum_test, input_uv_cumsum_test = utils_cummulative.cummulative_fn(depth=pred_depth_test, uv=uv_test, depth_teacher=gt_test_dict['o_with_f'][..., [0]], startpos=input_test_dict['startpos'], lengths=input_test_dict['lengths'], eot=pred_flag_test, cam_params_dict=cam_params_dict, epoch=0, args=args, gt=gt_test_dict['xyz'][..., [0, 1, 2]], bi_pred_weight=bi_pred_weight_test)

    # Project the (u, v, depth) to world space
    pred_xyz_test = pt.stack([utils_transform.projectToWorldSpace(uv=input_uv_cumsum_test[i], depth=pred_depth_cumsum_test[i], cam_params_dict=cam_params_dict, device=device) for i in range(input_uv_cumsum_test.shape[0])])

  if 'refinement' in args.pipeline and not args.error_analysis:
    pred_xyz_test = utils_model.refinement(model_dict=model_dict, gt_dict=gt_test_dict, cam_params_dict=cam_params_dict, pred_xyz=pred_xyz_test, optimize=args.optimize, pred_dict=pred_dict_test, missing_dict=missing_dict_test)

  elif args.error_analysis:
    pred_xyz_test_list = latent_error_analysis(model_dict=model_dict, gt_dict=gt_test_dict, cam_params_dict=cam_params_dict, pred_xyz=pred_xyz_test, optimize=args.optimize, pred_dict=pred_dict_test, input_dict=input_test_dict, missing_dict=missing_dict_test)

  # utils_func.print_loss(loss_list=[test_loss_dict, test_loss], name='Testing')

  if vis_flag == True and not args.error_analysis:
    pred_test_dict = {'input':in_test, 'flag':pred_flag_test, 'depth':pred_depth_test, 'xyz':pred_xyz_test}
    utils_inference_func.make_visualize(input_test_dict=input_test_dict, gt_test_dict=gt_test_dict, evaluation_results=evaluation_results, animation_visualize_flag=args.animation, pred_test_dict=pred_test_dict, visualization_path=visualization_path, args=args)


def calculate_optimization_loss(optimized_xyz, gt_dict, cam_params_dict):
  below_ground_loss = utils_loss.BelowGroundPenalize(pred=optimized_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
  trajectory_loss = utils_loss.TrajectoryLoss(pred=optimized_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
  gravity_loss = utils_loss.GravityLoss(pred=optimized_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
  if len(args.multiview_loss) > 0:
    multiview_loss = utils_loss.MultiviewReprojectionLoss(pred=optimized_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1]], lengths=gt_dict['lengths'], cam_params_dict=cam_params_dict)
  else:
    multiview_loss = pt.tensor(0.)

  # print(trajectory_loss)
  # print(below_ground_loss)
  # print(multiview_loss)
  optimization_loss = below_ground_loss + multiview_loss # + trajectory_loss # + multiview_loss #+ trajectory_loss # 
  return optimization_loss

def latent_error_analysis(model_dict, input_dict, gt_dict, cam_params_dict, pred_xyz, optimize, pred_dict, missing_dict):
  # Determine the latent size
  loss = []
  latent_size = 0
  if 'sin_cos' in args.latent_code:
    latent_size += 2
  if 'angle' in args.latent_code:
    latent_size += 1
  if 'f' in args.latent_code:
    latent_size += 3
  if 'f_norm' in args.latent_code:
    latent_size += 3

  ####################################
  ############ INPUT PREP ############
  ####################################
  if args.in_refine == 'xyz':
    # xyz
    in_f = pred_xyz
    # lengths
    lengths = gt_dict['lengths']
  elif args.in_refine == 'dtxyz':
    # dtxyz
    pred_xyz_delta = pred_xyz[:, :-1, :] - pred_xyz[:, 1:, :]
    in_f = pred_xyz_delta
    # lengths
    lengths = gt_dict['lengths']-1
  elif args.in_refine =='xyz_dtxyz':
    # dtxyz
    pred_xyz_delta = pred_xyz[:, :-1, :] - pred_xyz[:, 1:, :]
    pred_xyz_delta = pt.cat((pred_xyz_delta, pred_xyz_delta[:, [-1], :]), dim=1)
    pred_xyz_delta = utils_func.duplicate_at_length(seq=pred_xyz_delta, lengths=gt_dict['lengths'])
    # xyz & dtxyz & latent
    in_f = pt.cat((pred_xyz, pred_xyz_delta), dim=2)
    # lengths
    lengths = gt_dict['lengths']

    # Initialize
  latent_analyzer = OptimizationLatentAnalyze(model_dict=model_dict, pred_xyz=pred_xyz, gt_dict=gt_dict, cam_params_dict=cam_params_dict, latent_size=latent_size,
                                                            n_refinement=args.n_refinement, pred_dict=pred_dict, latent_code=args.latent_code)
  if 'sin_cos' in args.latent_code:
    degree = np.linspace(0, 360.0, int(1e2))
    radian = degree * np.pi / 180
    best = {'xyz':None, 'loss':None, 'rad':None}
    worst = {'xyz':None, 'loss':None, 'rad':None}
    for idx, each_rad in enumerate(tqdm(radian, desc='Latent : Sin, Cos')):
      rad = np.expand_dims(np.array([each_rad, each_rad]), axis=[0, 1])
      rad = pt.tensor(rad * np.ones((in_f.shape[0], in_f.shape[1], 2)))
      latent = pt.cat((pt.sin(rad[..., [0]]), pt.cos(rad[..., [1]])), dim=2)
      in_f_ = pt.cat((in_f.clone(), latent.type(pt.cuda.FloatTensor).to(device)), dim=2)
      pred_refinement = latent_analyzer(in_f=in_f_.detach().clone(), lengths=lengths)

      ####################################
      ########### OUTPUT PREP ############
      ####################################
      if args.out_refine == 'dtxyz_cumsum':
        # Cummulative from t=0
        if args.in_refine == 'xyz' or args.in_refine == 'xyz_dtxyz':
          pred_xyz_optimized = pt.cumsum(pt.cat((pred_xyz[:, [0], :].detach().clone(), pred_refinement[:, :-1, :]), dim=1), dim=1)
        elif args.in_refine == 'dtxyz':
          pred_xyz_optimized = pt.cumsum(pt.cat((pred_xyz[:, [0], :].detach().clone(), pred_refinement), dim=1), dim=1)

      elif args.out_refine == 'dtxyz_consec':
        # Consecutive from (t-1) + dt
        if args.in_refine == 'xyz' or args.in_refine == 'xyz_dtxyz':
          pred_xyz_optimized = pt.cat((pred_xyz[:, [0], :].detach().clone(), (pred_xyz[:, :-1, :].detach().clone() + pred_refinement[:, :-1, :])), dim=1)
        elif args.in_refine == 'dtxyz':
          pred_xyz_optimized = pt.cat((pred_xyz[:, [0], :].detach().clone(), (pred_xyz[:, :-1, :].detach().clone() + pred_refinement)), dim=1)

      elif args.out_refine == 'xyz':
        if args.in_refine == 'xyz' or args.in_refine == 'xyz_dtxyz':
          pred_xyz_optimized = pred_xyz + pred_refinement
        elif args.in_refine == 'dtxyz':
          pred_xyz_optimized = pt.cat((pred_xyz[:, [0], :].detach().clone(), pred_xyz[:, 1:, :].detach().clone() + pred_refinement), dim=1)

      test_loss_dict, test_loss = utils_model.calculate_loss(pred_xyz=pred_xyz_optimized[..., [0, 1, 2]], input_dict=input_dict, gt_dict=gt_dict, cam_params_dict=cam_params_dict, pred_dict=pred_dict, missing_dict=missing_dict)
      optimization_loss = calculate_optimization_loss(optimized_xyz=pred_xyz_optimized, gt_dict=gt_dict, cam_params_dict=cam_params_dict)
      test_loss = optimization_loss


      if idx == 0:
        best['loss'] = test_loss
        best['xyz'] = pred_xyz_optimized
        best['rad'] = each_rad

        worst['loss'] = test_loss
        worst['xyz'] = pred_xyz_optimized
        worst['rad'] = each_rad
      else:
        if (test_loss < best['loss']):
          best['loss'] = test_loss
          best['xyz'] = pred_xyz_optimized
          best['rad'] = each_rad
        if (test_loss > worst['loss']):
          worst['loss'] = test_loss
          worst['xyz'] = pred_xyz_optimized
          worst['rad'] = each_rad

      loss.append(test_loss.detach().cpu().numpy())


    fig = make_subplots(rows=2, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}], [{'colspan':2}, None]])
    # Best/Worst 3D trajaectory
    best_xyz = best['xyz'].cpu().detach().numpy()
    worst_xyz = worst['xyz'].cpu().detach().numpy()
    best_x, best_y, best_z = best_xyz[..., 0].reshape(-1), best_xyz[..., 1].reshape(-1), best_xyz[..., 2].reshape(-1)
    worst_x, worst_y, worst_z = worst_xyz[..., 0].reshape(-1), worst_xyz[..., 1].reshape(-1), worst_xyz[..., 2].reshape(-1)
    # GT
    gt_xyz = gt_dict['xyz'].cpu().detach().numpy()
    gt_x, gt_y, gt_z = gt_xyz[..., 0].reshape(-1), gt_xyz[..., 1].reshape(-1), gt_xyz[..., 2].reshape(-1)
    # Trajectory - Pred
    fig.add_trace(go.Scatter3d(x=best_x, y=best_y, z=-best_z, mode='markers+lines', marker=marker_dict_pred, name='Best Loss : {}, Rad : {}'.format(best['loss'], best['rad'])), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=worst_x, y=worst_y, z=-worst_z, mode='markers+lines', marker=marker_dict_pred, name='Worst Loss : {}, Rad : {}'.format(worst['loss'], worst['rad'])), row=1, col=2)
    # Trajectory - GT
    fig.add_trace(go.Scatter3d(x=gt_x, y=gt_y, z=-gt_z, mode='markers+lines', marker=marker_dict_gt), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=gt_x, y=gt_y, z=-gt_z, mode='markers+lines', marker=marker_dict_gt), row=1, col=2)
    # Latent 
    best_latent_sin, best_latent_cos = np.sin(best['rad']), np.cos(best['rad'])
    best_latent_arrow_x = np.array([best_x[0], best_x[0] + best_latent_sin * 3])
    best_latent_arrow_y = np.array([0, 0])
    best_latent_arrow_z = np.array([best_z[0], best_z[0] + best_latent_cos * 3])
    fig.add_trace(go.Scatter3d(x=best_latent_arrow_x, y=best_latent_arrow_y, z=-best_latent_arrow_z, mode='lines', line=dict(width=10), marker=marker_dict_latent), row=1, col=1)
    worst_latent_sin, worst_latent_cos = np.sin(worst['rad']), np.cos(worst['rad'])
    worst_latent_arrow_x = np.array([worst_x[0], worst_x[0] + worst_latent_sin * 3])
    worst_latent_arrow_y = np.array([0, 0])
    worst_latent_arrow_z = np.array([worst_z[0], worst_z[0] + worst_latent_cos * 3])
    fig.add_trace(go.Scatter3d(x=worst_latent_arrow_x, y=worst_latent_arrow_y, z=-worst_latent_arrow_z, mode='lines', line=dict(width=10), marker=marker_dict_latent), row=1, col=2)
    # Axis reference
    axis_offset = min([gt_x[0], gt_z[0]]) + 1
    selector = [[axis_offset, 0], [0, axis_offset]] # Draw x and z
    for i, sel in enumerate(selector):
      axis_x = np.array([gt_x[0], gt_x[0] + sel[0]])
      axis_y = np.array([0, 0])
      axis_z = np.array([gt_z[0], gt_z[0] + sel[1]])
      fig.add_trace(go.Scatter3d(x=axis_x, y=axis_y, z=-axis_z, mode='lines', line=dict(width=10), marker=marker_dict_axis[i]), row=1, col=1)
      fig.add_trace(go.Scatter3d(x=axis_x, y=axis_y, z=-axis_z, mode='lines', line=dict(width=10), marker=marker_dict_axis[i]), row=1, col=2)
    # Loss landscape
    fig.add_trace(go.Scatter(x=radian, y=loss, mode='markers+lines'), row=2, col=1)
    global batch_ptr
    fig['layout']['scene1'].update(xaxis=dict(dtick=1, range=[-4, 4],), yaxis = dict(dtick=1, range=[-4, 4],), zaxis = dict(dtick=1, range=[-4, 4]), aspectmode='manual', aspectratio=dict(x=1, y=1, z=1))
    fig['layout']['scene2'].update(xaxis=dict(dtick=1, range=[-4, 4],), yaxis = dict(dtick=1, range=[-4, 4],), zaxis = dict(dtick=1, range=[-4, 4]), aspectmode='manual', aspectratio=dict(x=1, y=1, z=1))
    plotly.offline.plot(fig, filename='./{}/Latent_Analysis_optimization_1e4/{}.html'.format(args.visualization_path, batch_ptr), auto_open=True)
    batch_ptr += 1


def collate_fn_padd(batch):
    # Padding batch of variable length
    # Columns convention : (x, y, z, u, v, d, eot, og, rad)
    padding_value = -10
    ## Get sequence lengths
    lengths = pt.tensor([trajectory[1:, :].shape[0] for trajectory in batch])
    # Input features : columns 4-5 contain u, v in screen space
    ## Padding
    input_batch = [pt.Tensor(trajectory[1:, input_col]) for trajectory in batch] # (4, 5, -2) = (u, v ,end_of_trajectory)
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=padding_value)
    ## Retrieve initial position (u, v, depth)
    input_startpos = pt.stack([pt.Tensor(trajectory[0, input_startpos_col]) for trajectory in batch])  # (4, 5, 6, -2) = (u, v, depth, end_of_trajectory)
    input_startpos = pt.unsqueeze(input_startpos, dim=1)
    ## Compute mask
    input_mask = (input_batch != padding_value)

    # Output features : columns 6 cotain depth from camera to projected screen
    ## Padding
    gt_batch = [pt.Tensor(trajectory[1:, gt_col]) for trajectory in batch]
    gt_batch = pad_sequence(gt_batch, batch_first=True)
    ## Retrieve initial position
    gt_startpos = pt.stack([pt.Tensor(trajectory[0, gt_startpos_col]) for trajectory in batch])
    gt_startpos = pt.unsqueeze(gt_startpos, dim=1)
    ## Retrieve the x, y, z in world space for compute the reprojection error (x', y', z' <===> x, y, z)
    ## Compute cummulative summation to form a trajectory from displacement every columns except the end_of_trajectory
    if args.optimize:
      gt_xyz = [pt.cat((pt.cumsum(pt.Tensor(trajectory[..., [x, y, z]]), dim=0), pt.Tensor(trajectory[..., :])), dim=-1) for trajectory in batch]
    else:
      gt_xyz = [pt.cat((pt.cumsum(pt.Tensor(trajectory[..., [x, y, z]]), dim=0), pt.Tensor(trajectory[..., features_cols])), dim=-1) for trajectory in batch]
    gt_xyz = pad_sequence(gt_xyz, batch_first=True, padding_value=padding_value)
    ## Compute mask
    gt_mask = (gt_xyz != padding_value)

    return {'input':[input_batch, lengths, input_mask, input_startpos],
            'gt':[gt_batch, lengths+1, gt_mask, gt_startpos, gt_xyz]}

if __name__ == '__main__':
  print('[#]Testing : Trajectory Estimation')

  # Initialize folder
  utils_func.initialize_folder(args.visualization_path)
  # Load camera parameters : ProjectionMatrix and WorldToCameraMatrix
  cam_params_dict = utils_transform.get_cam_params_dict(args.cam_params_file, device)

  # Create Datasetloader for test and testidation
  trajectory_test_dataset = TrajectoryDataset(dataset_path=args.dataset_test_path, trajectory_type=args.trajectory_type)
  trajectory_test_dataloader = DataLoader(trajectory_test_dataset, batch_size=args.batch_size, num_workers=10, shuffle=False, collate_fn=collate_fn_padd, pin_memory=True, drop_last=False)
  # Cast it to iterable object
  trajectory_test_iterloader = iter(trajectory_test_dataloader)

  # Dataset format
  # Trajectory path : (x0, y0) ... (xn, yn)
  print("======================================================Summary Batch (batch_size = {})=========================================================================".format(args.batch_size))
  for key, batch in enumerate(trajectory_test_dataloader):
    print("Input batch [{}] : batch={}, lengths={}, mask={}, initial position={}".format(key, batch['input'][0].shape, batch['input'][1].shape, batch['input'][2].shape, batch['input'][3].shape))
    print("gt batch [{}] : batch={}, lengths={}, mask={}, initial position={}".format(key, batch['gt'][0].shape, batch['gt'][1].shape, batch['gt'][2].shape, batch['gt'][3].shape))
    # Test RNN/LSTM Step
    # 1.Pack the padded
    packed = pt.nn.utils.rnn.pack_padded_sequence(batch['input'][0], batch_first=True, lengths=batch['input'][1], enforce_sorted=False)
    # 2.RNN/LSTM model
    # 3.Unpack the packed
    unpacked = pt.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, padding_value=-10)
    print("Unpacked equality : ", pt.eq(batch['input'][0], unpacked[0]).all())
    print("===============================================================================================================================================================")
  # Model definition
  model_dict, model_cfg = utils_func.get_model_depth(model_arch=args.model_arch, features_cols=features_cols, args=args)
  print(model_dict)
  print(model_cfg)
  model_dict = {model:model_dict[model].to(device) for model in model_dict.keys()}

  # Load the checkpoint if it's available.
  if args.load_checkpoint is None:
    # Create a model
    print('===>No model checkpoint')
    print('[#] Define the Learning rate, Optimizer, Decay rate and Scheduler...')
  else:
    print('===>Load checkpoint with Optimizer state, Decay and Scheduler state')
    print('[#] Loading ... {}'.format(args.load_checkpoint))
    model_dict = utils_func.load_checkpoint_predict(model_dict)

  print('[#]Model Architecture')
  for model in model_cfg.keys():
    print('####### Model - {} #######'.format(model))
    print(model_dict[model])

  # Test a model iterate over dataloader to get each batch and pass to predict function
  n_trajectory = 0
  for batch_idx, batch_test in enumerate(trajectory_test_dataloader):
    print("[#]Batch-{}".format(batch_idx))
    # Testing set (Each index in batch_test came from the collate_fn_padd)
    input_test_dict = {'input':batch_test['input'][0].to(device), 'lengths':batch_test['input'][1].to(device), 'mask':batch_test['input'][2].to(device), 'startpos':batch_test['input'][3].to(device)}
    gt_test_dict = {'o_with_f':batch_test['gt'][0].to(device), 'lengths':batch_test['gt'][1].to(device), 'mask':batch_test['gt'][2].to(device), 'startpos':batch_test['gt'][3].to(device), 'xyz':batch_test['gt'][4].to(device)}

    # Call function to test
    predict(input_test_dict=input_test_dict, gt_test_dict=gt_test_dict, model_dict=model_dict, vis_flag=args.vis_flag,
            threshold=args.threshold, cam_params_dict=cam_params_dict, visualization_path=args.visualization_path)

    n_trajectory += input_test_dict['input'].shape[0]

  print("[#] Done")
