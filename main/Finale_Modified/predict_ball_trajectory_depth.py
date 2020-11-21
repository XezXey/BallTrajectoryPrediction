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
parser.add_argument('--model_arch', dest='model_arch', type=str, help='Input the model architecture(lstm, bilstm, gru, bigru)', required=True)
parser.add_argument('--threshold', dest='threshold', type=float, help='Provide the error threshold of reconstructed trajectory', default=0.8)
parser.add_argument('--no_animation', dest='animation_visualize_flag', help='Animated visualize flag', action='store_false')
parser.add_argument('--animation', dest='animation_visualize_flag', help='Animated visualize flag', action='store_true')
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
parser.add_argument('--optimize', dest='optimize', help='Flag to optimze(This will work when train with latent', action='store_true', default=False)

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
features = ['x', 'y', 'z', 'u', 'v', 'd', 'eot', 'og', 'rad', 'f_sin', 'f_cos', 'g']
x, y, z, u, v, d, eot, og, rad, f_sin, f_cos, g = range(len(features))
input_col, input_startpos_col, gt_col, gt_startpos_col, gt_xyz_col, features_cols = utils_func.get_selected_cols(args=args, pred='depth')

def add_flag_noise(flag, lengths):
  flag = flag * 0.
  # for idx in range(flag.shape[0]):
    # print(flag)
    # flag_active = pt.where(flag[idx]==1.)
    # exit()
  return flag

def get_each_batch_trajectory(pred, gt, mask, lengths, cam_params_dict):
  gt_xyz = []
  pred_xyz = []
  gt_d = []
  pred_d = []

  _, _, pred_d_tmp = utils_transform.projectToScreenSpace(world=pred, cam_params_dict=cam_params_dict['main'], normalize=False)
  _, _, gt_d_tmp = utils_transform.projectToScreenSpace(world=gt, cam_params_dict=cam_params_dict['main'], normalize=False)
  mask_d = mask[..., [0]]

  for i in range(lengths.shape[0]):
    if i == 0:
      gt_xyz = ((gt[i] * mask[i])[:lengths[i], :])
      pred_xyz = ((pred[i] * mask[i])[:lengths[i], :])
      gt_d = ((gt_d_tmp[i] * mask_d[i])[:lengths[i], :])
      pred_d = ((pred_d_tmp[i] * mask_d[i])[:lengths[i], :])
    else:
      gt_xyz = pt.cat((gt_xyz, (gt[i] * mask[i])[:lengths[i], :]), dim=0)
      pred_xyz = pt.cat((pred_xyz, (pred[i] * mask[i])[:lengths[i], :]), dim=0)
      gt_d = pt.cat((gt_d, (gt_d_tmp[i] * mask_d[i])[:lengths[i], :]), dim=0)
      pred_d = pt.cat((pred_d, (pred_d_tmp[i] * mask_d[i])[:lengths[i], :]), dim=0)

  return {'pred_xyz':pred_xyz.cpu().detach(), 'gt_xyz':gt_xyz.cpu().detach(), 'pred_d':pred_d.cpu().detach(), 'gt_d':gt_d.cpu().detach()}

def evaluateModel(pred, gt, mask, lengths, cam_params_dict, threshold=1, delmask=True):
  # accepted_3axis_maxdist, accepted_3axis_loss, accepted_trajectory_loss, mae_loss_trajectory, mae_loss_3axis, maxdist_3axis, mse_loss_3axis
  evaluation_results = {'MAE':{}, 'MSE':{}, 'RMSE':{}}
  # metrics = ['3axis_maxdist', '3axis_loss', 'trajectory_loss', 'accepted_3axis_loss', 'accepted_3axis_maxdist', 'accepted_trajectory_loss']

  _, _, pred_d = utils_transform.projectToScreenSpace(world=pred, cam_params_dict=cam_params_dict['main'], normalize=False)
  _, _, gt_d = utils_transform.projectToScreenSpace(world=gt, cam_params_dict=cam_params_dict['main'], normalize=False)
  mask_d = mask[..., [0]]

  for distance in evaluation_results:
    if distance == 'MAE':
      loss_3axis = pt.sum(((pt.abs(gt - pred)) * mask), axis=1) / pt.sum(mask, axis=1)
      maxdist_3axis = pt.max(pt.abs(gt - pred) * mask, dim=1)[0]
      loss_depth = pt.sum(((pt.abs(gt_d - pred_d)) * mask_d), axis=1) / pt.sum(mask_d, axis=1)
      maxdist_depth = pt.sum(((pt.abs(gt_d - pred_d)) * mask_d), axis=1) / pt.sum(mask_d, axis=1)
    elif distance == 'MSE':
      loss_3axis = pt.sum((((gt - pred)**2) * mask), axis=1) / pt.sum(mask, axis=1)
      maxdist_3axis = pt.max(((gt - pred)**2) * mask, dim=1)[0]
      loss_depth = pt.sum((((gt_d - pred_d)**2) * mask_d), axis=1) / pt.sum(mask_d, axis=1)
      maxdist_depth = pt.sum(((pt.abs(gt_d - pred_d)**2) * mask_d), axis=1) / pt.sum(mask_d, axis=1)
    elif distance == 'RMSE':
      loss_3axis = pt.sqrt(pt.sum((((gt - pred)**2) * mask), axis=1) / pt.sum(mask, axis=1))
      maxdist_3axis = pt.max(((gt - pred)**2) * mask, dim=1)[0]
      loss_depth = pt.sqrt(pt.sum(((pt.abs(gt_d - pred_d)**2) * mask_d), axis=1) / pt.sum(mask_d, axis=1))
      maxdist_depth = pt.sum(((pt.abs(gt_d - pred_d)**2) * mask_d), axis=1) / pt.sum(mask_d, axis=1)

    # Trajectory 3 axis loss
    evaluation_results[distance]['maxdist_3axis'] = maxdist_3axis.cpu().detach().numpy()
    evaluation_results[distance]['loss_3axis'] = loss_3axis.cpu().detach().numpy()
    evaluation_results[distance]['mean_loss_3axis'] = pt.mean(loss_3axis, axis=0).cpu().detach().numpy()
    evaluation_results[distance]['sd_loss_3axis'] = pt.std(loss_3axis, axis=0).cpu().detach().numpy()
    # Depth loss
    evaluation_results[distance]['loss_depth'] = loss_depth.cpu().detach().numpy()
    evaluation_results[distance]['mean_loss_depth'] = pt.mean(loss_depth, axis=0).cpu().detach().numpy()
    evaluation_results[distance]['sd_loss_depth'] = pt.std(loss_depth, axis=0).cpu().detach().numpy()
    # Accepted Trajectory below threshold
    evaluation_results[distance]['accepted_3axis_loss'] = pt.sum((pt.sum(loss_3axis < threshold, axis=1) == 3)).cpu().detach().numpy()
    evaluation_results[distance]['accepted_3axis_maxdist']= pt.sum((pt.sum(maxdist_3axis < threshold, axis=1) == 3)).cpu().detach().numpy()

    print("Distance : ", distance)
    print("Accepted 3-Axis(X, Y, Z) Maxdist < {} : {}".format(threshold, evaluation_results[distance]['accepted_3axis_maxdist']))
    print("Accepted 3-Axis(X, Y, Z) loss < {} : {}".format(threshold, evaluation_results[distance]['accepted_3axis_loss']))

  # Accepted trajectory < Threshold
  return evaluation_results

def evaluate(all_batch_trajectory):
  print("[#]Summary All Trajectory")
  distance = ['MAE', 'MSE', 'RMSE']
  space = ['xyz', 'd']
  trajectory = {}
  for key in all_batch_trajectory.keys():
    if len(all_batch_trajectory[key]) == 0:
      print("Skipping key=[{}]".format(key))
      trajectory[key] = []
      continue
    trajectory[key] = pt.cat((all_batch_trajectory[key]), dim=0)

  gt_xyz = trajectory['gt_xyz']
  pred_xyz = trajectory['pred_xyz']
  gt_d = trajectory['gt_d']
  pred_d = trajectory['pred_d']

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

  pred_dict_test, in_test = utils_model.fw_pass(model_dict, input_dict=input_test_dict, cam_params_dict=cam_params_dict)

  pred_depth_test, pred_flag_test, input_flag_test = utils_func.get_pipeline_var(pred_dict=pred_dict_test, input_dict=input_test_dict)

  if args.bi_pred_weight:
    bi_pred_weight_test = pred_depth_test[..., [2]]
  else:
    bi_pred_weight_test = pt.zeros(pred_depth_test[..., [0]].shape)

  pred_depth_cumsum_test, input_uv_cumsum_test = utils_cummulative.cummulative_fn(depth=pred_depth_test, uv=input_test_dict['input'][..., [0, 1]], depth_teacher=gt_test_dict['o_with_f'][..., [0]], startpos=input_test_dict['startpos'], lengths=input_test_dict['lengths'], eot=pred_flag_test, cam_params_dict=cam_params_dict, epoch=0, args=args, gt=gt_test_dict['xyz'][..., [0, 1, 2]], bi_pred_weight=bi_pred_weight_test)

  # Project the (u, v, depth) to world space
  pred_xyz_test = pt.stack([utils_transform.projectToWorldSpace(uv=input_uv_cumsum_test[i], depth=pred_depth_cumsum_test[i], cam_params_dict=cam_params_dict, device=device) for i in range(input_uv_cumsum_test.shape[0])])

  if 'refinement' in args.pipeline:
      pred_xyz_test = utils_model.refinement(model_dict=model_dict, gt_dict=gt_test_dict, cam_params_dict=cam_params_dict, pred_xyz=pred_xyz_test, optimize=args.optimize)

  test_loss_dict, test_loss = utils_model.calculate_loss(pred_xyz=pred_xyz_test, input_dict=input_test_dict, gt_dict=gt_test_dict, cam_params_dict=cam_params_dict, pred_dict=pred_dict_test)

  ####################################
  ############# Evaluation ###########
  ####################################
  # Calculate loss per trajectory
  evaluation_results = evaluateModel(pred=pred_xyz_test, gt=gt_test_dict['xyz'][..., [0, 1, 2]], mask=gt_test_dict['mask'][..., [0, 1, 2]], lengths=gt_test_dict['lengths'], threshold=threshold, cam_params_dict=cam_params_dict)
  reconstructed_trajectory = [gt_test_dict['xyz'][..., [0, 1, 2]].detach().cpu().numpy(), pred_xyz_test.detach().cpu().numpy(), input_uv_cumsum_test.detach().cpu().numpy(), pred_depth_cumsum_test.detach().cpu().numpy(), gt_test_dict['lengths'].detach().cpu().numpy()]
  each_batch_trajectory = get_each_batch_trajectory(pred=pred_xyz_test, gt=gt_test_dict['xyz'][..., [0, 1, 2]], mask=gt_test_dict['mask'][..., [0, 1, 2]], lengths=gt_test_dict['lengths'], cam_params_dict=cam_params_dict)

  utils_func.print_loss(loss_list=[test_loss_dict, test_loss], name='Testing')

  if vis_flag == True:
    pred_test_dict = {'input':in_test, 'flag':pred_flag_test, 'depth':pred_depth_test, 'xyz':pred_xyz_test}
    utils_inference_func.make_visualize(input_test_dict=input_test_dict, gt_test_dict=gt_test_dict, evaluation_results=evaluation_results, animation_visualize_flag=args.animation_visualize_flag, pred_test_dict=pred_test_dict, visualization_path=visualization_path, args=args)

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
    gt_xyz = [pt.cat((pt.cumsum(pt.Tensor(trajectory[..., [x, y, z]]), dim=0), pt.Tensor(trajectory[..., features_cols])), dim=-1) for trajectory in batch]
    gt_xyz = pad_sequence(gt_xyz, batch_first=True, padding_value=padding_value)
    ## Compute mask
    gt_mask = (gt_xyz != padding_value)

    return {'input':[input_batch, lengths, input_mask, input_startpos],
            'gt':[gt_batch, lengths+1, gt_mask, gt_startpos, gt_xyz]}

def load_checkpoint(model_dict):
  print("="*100)
  print("[#] Model Parameters")
  for model in model_dict.keys():
    for k, v in model_dict[model].named_parameters():
      print("===> ", k, v.shape)
  print("="*100)
  if os.path.isfile(args.load_checkpoint):
    print("[#] Found the checkpoint...")
    checkpoint = pt.load(args.load_checkpoint, map_location='cuda:0')
    # Load optimizer, learning rate, decay and scheduler parameters
    print(checkpoint['model_cfg'].keys())
    for model in checkpoint['model_cfg'].keys():
      if 'model' in model:
        available_module = model.split('_')[1]
        if available_module == 'flag':
          available_module = 'eot'
        if available_module in args.pipeline:
          model_dict[model].load_state_dict(checkpoint[model])
      else:
        model_dict['model_{}'.format(model)].load_state_dict(checkpoint['model_{}'.format(model)])
    return model_dict

  else:
    print("[#] Checkpoint not found...")
    exit()

def summary(evaluation_results_all):
  print("="*100)
  summary_evaluation = evaluation_results_all[0]
  print("[#]Summary")
  for distance in summary_evaluation.keys():
    for idx, each_batch_eval in enumerate(evaluation_results_all):
      if idx == 0:
        continue
      summary_evaluation[distance]['maxdist_3axis'] = np.concatenate((summary_evaluation[distance]['maxdist_3axis'], each_batch_eval[distance]['maxdist_3axis']), axis=0)
      summary_evaluation[distance]['loss_3axis'] = np.concatenate((summary_evaluation[distance]['loss_3axis'], each_batch_eval[distance]['loss_3axis']), axis=0)
      summary_evaluation[distance]['loss_depth'] = np.concatenate((summary_evaluation[distance]['loss_depth'], each_batch_eval[distance]['loss_depth']), axis=0)
      summary_evaluation[distance]['accepted_3axis_loss'] += each_batch_eval[distance]['accepted_3axis_loss']
      summary_evaluation[distance]['accepted_3axis_maxdist'] += each_batch_eval[distance]['accepted_3axis_maxdist']

    print("Distance : ", distance)
    print("Mean 3-Axis(X, Y, Z) loss : {}".format(np.mean(summary_evaluation[distance]['loss_3axis'], axis=0)))
    print("SD 3-Axis(X, Y, Z) loss : {}".format(np.std(summary_evaluation[distance]['loss_3axis'], axis=0)))
    print("Mean Depth loss : {}".format(np.mean(summary_evaluation[distance]['loss_depth'], axis=0)))
    print("SD Depth loss : {}".format(np.std(summary_evaluation[distance]['loss_depth'], axis=0)))
    print("Accepted trajectory by 3axis Loss : {} from {}".format(summary_evaluation[distance]['accepted_3axis_loss'], n_trajectory))
    print("Accepted trajectory by 3axis MaxDist : {} from {}".format(summary_evaluation[distance]['accepted_3axis_maxdist'], n_trajectory))

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
    model_dict = load_checkpoint(model_dict)

  print('[#]Model Architecture')
  for model in model_cfg.keys():
    print('####### Model - {} #######'.format(model))
    print(model_dict[model])

  # Test a model iterate over dataloader to get each batch and pass to predict function
  evaluation_results_all = []
  reconstructed_trajectory_all = []
  all_batch_trajectory = {'gt_xyz':[], 'pred_xyz':[], 'gt_d':[], 'pred_d':[]}
  n_trajectory = 0
  for batch_idx, batch_test in enumerate(trajectory_test_dataloader):
    print("[#]Batch-{}".format(batch_idx))
    # Testing set (Each index in batch_test came from the collate_fn_padd)
    input_test_dict = {'input':batch_test['input'][0].to(device), 'lengths':batch_test['input'][1].to(device), 'mask':batch_test['input'][2].to(device), 'startpos':batch_test['input'][3].to(device)}
    gt_test_dict = {'o_with_f':batch_test['gt'][0].to(device), 'lengths':batch_test['gt'][1].to(device), 'mask':batch_test['gt'][2].to(device), 'startpos':batch_test['gt'][3].to(device), 'xyz':batch_test['gt'][4].to(device)}

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
  # Save prediction file
  if args.savetofile is not None:
    utils_func.initialize_folder(args.savetofile)
    utils_func.save_reconstructed(eval_metrics=summary_evaluation, trajectory=reconstructed_trajectory_all)
  print("[#] Done")
