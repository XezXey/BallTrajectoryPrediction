from __future__ import print_function
# Import libs
import numpy as np
np.set_printoptions(precision=4)
import torch as pt
pt.manual_seed(25)
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
parser.add_argument('--model_arch', dest='model_arch', help='Input the model architecture(lstm, bilstm, gru, bigru)', nargs='+', default=[])
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
parser.add_argument('--bidirectional', dest='bidirectional', help='Bidirectional', nargs='+', default=[])
parser.add_argument('--pred_uv_space', dest='pred_uv_space', help='Prediction space for uv interpolation', type=str, default='pixel')
parser.add_argument('--trainable_init', help='Trainable initial state', action='store_true', default=False)
parser.add_argument('--savetofile', dest='savetofile', help='Save the prediction trajectory for doing optimization', type=str, default=None)
parser.add_argument('--multiview_loss', dest='multiview_loss', help='Use multiview loss', nargs='+', default=[])
parser.add_argument('--round', dest='round', help='Rounding pixel', action='store_true', default=False)
parser.add_argument('--pipeline', dest='pipeline', help='Pipeline', nargs='+', default=[])
parser.add_argument('--n_refinement', dest='n_refinement', help='Refinement Iterations', type=int, default=1)
parser.add_argument('--fix_refinement', dest='fix_refinement', help='Fix Refinement for 1st and last points', action='store_true', default=False)
parser.add_argument('--optimize', dest='optimize', help='Flag to optimze(This will work when train with latent', default=None)
parser.add_argument('--latent_code', dest='latent_code', help='Optimze the latent code)', nargs='+', default=[])
parser.add_argument('--missing', dest='missing', help='Adding a missing data points while training', default=None)
parser.add_argument('--recon', dest='recon', help='Using Ideal or Noisy uv for reconstruction', default='ideal_uv')
parser.add_argument('--in_refine', dest='in_refine', help='Input of Refinement space', default='xyz')
parser.add_argument('--out_refine', dest='out_refine', help='Output of Refinement space', default='xyz')
parser.add_argument('--autoregressive', dest='autoregressive', help='Doing auto regression for interpolation', action='store_true', default=False)
parser.add_argument('--annealing', dest='annealing', help='Apply annealing', action='store_true', default=False)
parser.add_argument('--annealing_cycle', dest='annealing_cycle', type=int, help='Apply annealing every n epochs', default=5)
parser.add_argument('--annealing_gamma', dest='annealing_gamma', type=float, help='Apply annealing every n epochs', default=0.95)
parser.add_argument('--latent_transf', dest='latent_transf', type=str, help='Extra latent manipulation method', default=None)
parser.add_argument('--lr', dest='lr', type=float, help='Learning rate for optimizaer', default=10)
parser.add_argument('--load_missing', dest='load_missing', help='Use a missing annotation from a file', action='store_true', default=False)

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

def get_each_batch_trajectory(pred, gt, mask, lengths, cam_params_dict):
  gt_uv = []
  pred_uv = []

  for i in range(lengths.shape[0]):
    if i == 0:
      gt_uv = ((gt[i] * mask[i])[:lengths[i], :])
      pred_uv = ((pred[i] * mask[i])[:lengths[i], :])
    else:
      gt_uv = pt.cat((gt_uv, (gt[i] * mask[i])[:lengths[i], :]), dim=0)
      pred_uv = pt.cat((pred_uv, (pred[i] * mask[i])[:lengths[i], :]), dim=0)

  return {'pred_uv':pred_uv.cpu().detach(), 'gt_uv':gt_uv.cpu().detach()}

def evaluateModel(pred, gt, mask, lengths, cam_params_dict, threshold=1, delmask=True):
  # accepted_uv_maxdist, accepted_uv_loss, accepted_trajectory_loss, mae_loss_trajectory, mae_loss_uv, maxdist_uv, mse_loss_uv
  evaluation_results = {'MAE':{}, 'MSE':{}, 'RMSE':{}}
  # metrics = ['uv_maxdist', 'uv_loss', 'trajectory_loss', 'accepted_uv_loss', 'accepted_uv_maxdist', 'accepted_trajectory_loss']

  mask_d = mask[..., [0]]

  for distance in evaluation_results:
    if distance == 'MAE':
      loss_uv = pt.sum(((pt.abs(gt - pred)) * mask), axis=1) / pt.sum(mask, axis=1)
      maxdist_uv = pt.max(pt.abs(gt - pred) * mask, dim=1)[0]
    elif distance == 'MSE':
      loss_uv = pt.sum((((gt - pred)**2) * mask), axis=1) / pt.sum(mask, axis=1)
      maxdist_uv = pt.max(((gt - pred)**2) * mask, dim=1)[0]
    elif distance == 'RMSE':
      loss_uv = pt.sqrt(pt.sum((((gt - pred)**2) * mask), axis=1) / pt.sum(mask, axis=1))
      maxdist_uv = pt.max(((gt - pred)**2) * mask, dim=1)[0]

    # Trajectory 3 axis loss
    evaluation_results[distance]['maxdist_uv'] = maxdist_uv.cpu().detach().numpy()
    evaluation_results[distance]['loss_uv'] = loss_uv.cpu().detach().numpy()
    evaluation_results[distance]['mean_loss_uv'] = pt.mean(loss_uv, axis=0).cpu().detach().numpy()
    evaluation_results[distance]['sd_loss_uv'] = pt.std(loss_uv, axis=0).cpu().detach().numpy()
    # Accepted Trajectory below threshold
    evaluation_results[distance]['accepted_uv_loss'] = pt.sum((pt.sum(loss_uv < threshold, axis=1) == 3)).cpu().detach().numpy()
    evaluation_results[distance]['accepted_uv_maxdist']= pt.sum((pt.sum(maxdist_uv < threshold, axis=1) == 3)).cpu().detach().numpy()

    print("Distance : ", distance)
    print("Accepted 3-Axis(X, Y, Z) Maxdist < {} : {}".format(threshold, evaluation_results[distance]['accepted_uv_maxdist']))
    print("Accepted 3-Axis(X, Y, Z) loss < {} : {}".format(threshold, evaluation_results[distance]['accepted_uv_loss']))

  # Accepted trajectory < Threshold
  return evaluation_results

def evaluate(all_batch_trajectory):
  print("[#]Summary All Trajectory")
  distance = ['MAE', 'MSE', 'RMSE']
  space = ['uv']
  trajectory = {}
  for key in all_batch_trajectory.keys():
    if len(all_batch_trajectory[key]) == 0:
      print("Skipping key=[{}]".format(key))
      trajectory[key] = []
      continue
    trajectory[key] = pt.cat((all_batch_trajectory[key]), dim=0)

  gt_xyz = trajectory['gt_uv']
  pred_xyz = trajectory['pred_uv']

  for each_space in space:
    print("Space : ", each_space)
    gt = trajectory['gt_{}'.format(each_space)]
    pred = trajectory['pred_{}'.format(each_space)]
    for each_distance in distance:
      print("===>Distance : ", each_distance)
      if each_distance == 'MAE':
        mean = pt.mean((pt.abs(gt - pred)), dim=0)
        std = pt.std((pt.abs(gt - pred)), dim=0)
        print("MEAN : ", mean.cpu().detach().numpy())
        print("SD : ", std.cpu().detach().numpy())
      elif each_distance == 'MSE':
        mean = pt.mean(((gt - pred)**2), dim=0)
        std = pt.std(((gt - pred)**2), dim=0)
        print("MEAN : ", mean.cpu().detach().numpy())
        print("SD : ", std.cpu().detach().numpy())
      elif each_distance == 'RMSE':
        rmse = pt.sqrt(pt.mean(((gt - pred)**2), dim=0))
        print("RMSE : ", rmse.cpu().detach().numpy())
    print("*"*100)

def predict(input_test_dict, gt_test_dict, model_dict, threshold, cam_params_dict, vis_flag=True, visualization_path='./visualize_html/'):
  # Testing RNN/LSTM model
  # Run over each example
  # Test a model

  pred_dict_test = {}
  missing_dict_test = {}
  latent_optimized = []
  ####################################
  ############# Testing ##############
  ####################################
  # Evaluating mode
  # utils_model.eval_mode(model_dict=model_dict)

  # Forward pass
  pred_dict_test, in_test, missing_dict_test = utils_model.fw_pass(model_dict, input_dict=input_test_dict, cam_params_dict=cam_params_dict)
  # Pipeline variable
  pred_depth_test, pred_flag_test, input_flag_test = utils_func.get_pipeline_var(pred_dict=pred_dict_test, input_dict=input_test_dict)

  uv_test = utils_func.select_uv_recon(input_dict=input_test_dict, pred_dict=pred_dict_test, in_f_noisy=in_test)

  # Pipeline variable
  pred_depth_test, pred_flag_test, input_flag_test = utils_func.get_pipeline_var(pred_dict=pred_dict_test, input_dict=input_test_dict)

  pred_xyz_test = None

  # Calculate loss
  test_loss_dict, test_loss = utils_model.calculate_loss(pred_xyz=pred_xyz_test, input_dict=input_test_dict, gt_dict=gt_test_dict, cam_params_dict=cam_params_dict, pred_dict=pred_dict_test, missing_dict=missing_dict_test)

  ####################################
  ############# Evaluation ###########
  ####################################
  pred_uv = pred_dict_test['model_uv']
  # Calculate loss per trajectory (UV-interpolation in this case)

  utils_func.print_loss(loss_list=[test_loss_dict, test_loss], name='Testing')

  if vis_flag == True:
    pred_test_dict = {'input':in_test, 'flag':pred_flag_test, 'depth':pred_depth_test, 'xyz':pred_xyz_test, 'latent_optimized':latent_optimized}
    utils_inference_func.autoregressive_plot(pred_uv=pred_uv, gt_dict=input_test_dict)

  pred_uv = pt.cumsum(pt.cat((input_test_dict['startpos'][..., [0, 1]], pred_dict_test['model_uv']), dim=1), dim=1)
  gt_uv = pt.cumsum(pt.cat((input_test_dict['startpos'][..., [0, 1]], in_test[..., [0, 1]]), dim=1), dim=1)

  evaluation_results = evaluateModel(pred=pred_uv, gt=gt_uv, mask=gt_test_dict['mask'][..., [0, 1]], lengths=gt_test_dict['lengths'], threshold=threshold, cam_params_dict=cam_params_dict)
  reconstructed_trajectory = [gt_uv.detach().cpu().numpy(), pred_uv.detach().cpu().numpy(), gt_test_dict['lengths'].detach().cpu().numpy()]
  each_batch_trajectory = get_each_batch_trajectory(pred=pred_uv, gt=gt_uv, mask=gt_test_dict['mask'][..., [0, 1,]], lengths=gt_test_dict['lengths'], cam_params_dict=cam_params_dict)

  return evaluation_results, reconstructed_trajectory, each_batch_trajectory

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

    missing_mask = pt.tensor([trajectory[:, [-1]] for trajectory in batch])

    return {'input':[input_batch, lengths, input_mask, input_startpos, missing_mask],
            'gt':[gt_batch, lengths+1, gt_mask, gt_startpos, gt_xyz, missing_mask]}

def summary(evaluation_results_all):
  print("="*100)
  summary_evaluation = evaluation_results_all[0]
  print("[#]Summary")
  for distance in summary_evaluation.keys():
    for idx, each_batch_eval in enumerate(evaluation_results_all):
      if idx == 0:
        continue
      summary_evaluation[distance]['maxdist_uv'] = np.concatenate((summary_evaluation[distance]['maxdist_uv'], each_batch_eval[distance]['maxdist_uv']), axis=0)
      summary_evaluation[distance]['loss_uv'] = np.concatenate((summary_evaluation[distance]['loss_uv'], each_batch_eval[distance]['loss_uv']), axis=0)
      summary_evaluation[distance]['accepted_uv_loss'] += each_batch_eval[distance]['accepted_uv_loss']
      summary_evaluation[distance]['accepted_uv_maxdist'] += each_batch_eval[distance]['accepted_uv_maxdist']

    print("Distance : ", distance)
    print("Mean 3-Axis(X, Y, Z) loss : {}".format(np.mean(summary_evaluation[distance]['loss_uv'], axis=0)))
    print("SD 3-Axis(X, Y, Z) loss : {}".format(np.std(summary_evaluation[distance]['loss_uv'], axis=0)))
    print("Accepted trajectory by uv Loss : {} from {}".format(summary_evaluation[distance]['accepted_uv_loss'], n_trajectory))
    print("Accepted trajectory by uv MaxDist : {} from {}".format(summary_evaluation[distance]['accepted_uv_maxdist'], n_trajectory))

    print("="*100)

  return summary_evaluation

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
  evaluation_results_all = []
  reconstructed_trajectory_all = []
  all_batch_trajectory = {'gt_uv':[], 'pred_uv':[]}
  n_trajectory = 0
  for batch_idx, batch_test in enumerate(trajectory_test_dataloader):
    print("[#]Batch-{}".format(batch_idx))
    # Testing set (Each index in batch_test came from the collate_fn_padd)
    input_test_dict = {'input':batch_test['input'][0].to(device), 'lengths':batch_test['input'][1].to(device), 'mask':batch_test['input'][2].to(device), 'startpos':batch_test['input'][3].to(device), 'missing_mask':batch_test['input'][4].to(device)}
    gt_test_dict = {'o_with_f':batch_test['gt'][0].to(device), 'lengths':batch_test['gt'][1].to(device), 'mask':batch_test['gt'][2].to(device), 'startpos':batch_test['gt'][3].to(device), 'xyz':batch_test['gt'][4].to(device), 'missing_mask':batch_test['gt'][5].to(device)}

    # Call function to test
    evaluation_results, reconstructed_trajectory, each_batch_trajectory = predict(input_test_dict=input_test_dict, gt_test_dict=gt_test_dict,
                                                                                  model_dict=model_dict, vis_flag=args.vis_flag,
                                                                                  threshold=args.threshold, cam_params_dict=cam_params_dict, visualization_path=args.visualization_path)


    reconstructed_trajectory_all.append(reconstructed_trajectory)
    evaluation_results_all.append(evaluation_results)
    n_trajectory += input_test_dict['input'].shape[0]
    for key in each_batch_trajectory.keys():
      all_batch_trajectory[key].append(each_batch_trajectory[key])

  summary_evaluation = summary(evaluation_results_all)
  evaluate(all_batch_trajectory)
  # Save autoregression file for further prediction
  if args.autoregressive:
    utils_func.initialize_folder(args.savetofile)
    utils_func.save_autoregression(uv_interpolated=all_batch_trajectory, trajectory_loader=trajectory_test_dataloader)

  # Save prediction file
  # if args.savetofile is not None:
    # utils_func.initialize_folder(args.savetofile)
    # utils_func.save_reconstructed(eval_metrics=summary_evaluation, trajectory=reconstructed_trajectory_all)
  print("[#] Done")
