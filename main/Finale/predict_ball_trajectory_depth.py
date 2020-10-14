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
# Loss
import loss

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
parser.add_argument('--selected_features', dest='selected_features', help='Specify the selected features columns(eot, og, ', nargs='+', required=True)
parser.add_argument('--bi_pred', help='Bidirectional prediction', action='store_true', default=False)
parser.add_argument('--bi_pred_weight', help='Bidirectional prediction with weight', action='store_true', default=False)
parser.add_argument('--bw_pred', help='Backward prediction', action='store_true', default=False)
parser.add_argument('--env', dest='env', help='Environment', type=str, default='unity')
parser.add_argument('--savetofile', dest='savetofile', help='Save the prediction trajectory for doing optimization', type=str, default=None)

args = parser.parse_args()
utils_func.share_args(a=args)

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

def add_noise(input_trajectory, startpos, lengths):
  factor = np.random.uniform(low=0.6, high=0.95)
  if args.noise_sd is None:
    noise_sd = np.random.uniform(low=0.3, high=0.7)
  else:
    noise_sd = args.noise_sd

  input_trajectory = pt.cat((startpos, input_trajectory), dim=1)
  input_trajectory = pt.cumsum(input_trajectory, dim=1)
  # print(input_trajectory.shape)
  noise_uv = pt.normal(mean=0.0, std=noise_sd, size=input_trajectory[..., [0, 1]].shape).to(device)
  masking_noise = pt.nn.init.uniform_(pt.empty(input_trajectory[..., [0, 1]].shape)).to(device) > np.random.rand(1)[0]
  n_noise = int(input_trajectory.shape[0] * factor)
  noise_idx = np.random.choice(a=input_trajectory.shape[0], size=(n_noise,), replace=False)
  input_trajectory[noise_idx, :, :] += noise_uv[noise_idx, :, :] * masking_noise[noise_idx, :, :]
  input_trajectory = pt.tensor(np.diff(input_trajectory.cpu().numpy(), axis=1)).to(device)
  return input_trajectory

def add_flag_noise(flag, lengths):
  flag = flag * 0.
  # for idx in range(flag.shape[0]):
    # print(flag)
    # flag_active = pt.where(flag[idx]==1.)
    # exit()
  return flag

def evaluateModel(pred, gt, mask, lengths, threshold=1, delmask=True):
  # accepted_3axis_maxdist, accepted_3axis_loss, accepted_trajectory_loss, mae_loss_trajectory, mae_loss_3axis, maxdist_3axis, mse_loss_3axis
  evaluation_results = {'MAE':{}, 'MSE':{}}
  # metrics = ['3axis_maxdist', '3axis_loss', 'trajectory_loss', 'accepted_3axis_loss', 'accepted_3axis_maxdist', 'accepted_trajectory_loss']

  for distance in evaluation_results:
    if distance == 'MAE':
      loss_3axis = pt.sum(((pt.abs(gt - pred)) * mask), axis=1) / pt.sum(mask, axis=1)
      maxdist_3axis = pt.max(pt.abs(gt - pred) * mask, dim=1)[0]
    elif distance == 'MSE':
      loss_3axis = pt.sum((((gt - pred)**2) * mask), axis=1) / pt.sum(mask, axis=1)
      maxdist_3axis = pt.max(((gt - pred)**2) * mask, dim=1)[0]

    evaluation_results[distance]['maxdist_3axis'] = maxdist_3axis.cpu().detach().numpy()
    evaluation_results[distance]['loss_3axis'] = loss_3axis.cpu().detach().numpy()
    evaluation_results[distance]['mean_loss_3axis'] = pt.mean(loss_3axis, axis=0).cpu().detach().numpy()
    evaluation_results[distance]['sd_loss_3axis'] = pt.std(loss_3axis, axis=0).cpu().detach().numpy()
    evaluation_results[distance]['accepted_3axis_loss'] = pt.sum((pt.sum(loss_3axis < threshold, axis=1) == 3)).cpu().detach().numpy()
    evaluation_results[distance]['accepted_3axis_maxdist']= pt.sum((pt.sum(maxdist_3axis < threshold, axis=1) == 3)).cpu().detach().numpy()

    print("Distance : ", distance)
    print("Accepted 3-Axis(X, Y, Z) Maxdist < {} : {}".format(threshold, evaluation_results[distance]['accepted_3axis_maxdist']))
    print("Accepted 3-Axis(X, Y, Z) loss < {} : {}".format(threshold, evaluation_results[distance]['accepted_3axis_loss']))

  # Accepted trajectory < Threshold
  return evaluation_results

def predict(input_test_dict, gt_test_dict, model_flag, model_depth, threshold, cam_params_dict, vis_flag=True, visualization_path='./visualize_html/'):
  # Testing RNN/LSTM model
  # Run over each example
  # Test a model
  # Initial the state for model and Discriminator for EOT and Depth
  hidden_eot = model_flag.initHidden(batch_size=args.batch_size)
  cell_state_eot = model_flag.initCellState(batch_size=args.batch_size)
  hidden_depth = model_depth.initHidden(batch_size=args.batch_size)
  cell_state_depth = model_depth.initCellState(batch_size=args.batch_size)

  ####################################
  ############# Testing ##############
  ####################################
  # Training mode
  model_flag.eval()
  model_depth.eval()

  # Add noise on the fly
  in_test = input_test_dict['input'][..., [0, 1]].clone()
  if args.noise:
    in_test = add_noise(input_trajectory=in_test[..., [0, 1]].clone(), startpos=input_test_dict['startpos'][..., [0, 1]], lengths=input_test_dict['lengths'])

  ####################################
  ################ EOT ###############
  ####################################
  pred_eot_test, (_, _) = model_flag(in_test, hidden_eot, cell_state_eot, lengths=input_test_dict['lengths'])
  if args.flag_noise:
    pred_eot_test = add_flag_noise(flag=pred_eot_test, lengths=input_test_dict['input'])

  ####################################
  ############### Depth ##############
  ####################################
  in_test = pt.cat((in_test, pred_eot_test, input_test_dict['input'][..., 3:]), dim=2)  # Concat the (u_noise, v_noise, pred_eot, other_features(col index 3+)
  pred_depth_test, (_, _) = model_depth(in_test, hidden_depth, cell_state_depth, lengths=input_test_dict['lengths'])
  if args.bi_pred_weight:
    bi_pred_weight_test = pred_depth_test[..., [2]]
  else:
    bi_pred_weight_test = None

  pred_depth_cumsum_test, input_uv_cumsum_test = utils_cummulative.cummulative_fn(depth=pred_depth_test, uv=input_test_dict['input'][..., [0, 1]], depth_teacher=gt_test_dict['o_with_f'][..., [0]], startpos=input_test_dict['startpos'], lengths=input_test_dict['lengths'], eot=pred_eot_test, cam_params_dict=cam_params_dict, epoch=0, args=args, gt=gt_test_dict['xyz'][..., [0, 1, 2]], bi_pred_weight=bi_pred_weight_test)

  # Project the (u, v, depth) to world space
  pred_xyz_test = pt.stack([utils_transform.projectToWorldSpace(uv=input_uv_cumsum_test[i], depth=pred_depth_cumsum_test[i], cam_params_dict=cam_params_dict, device=device) for i in range(input_uv_cumsum_test.shape[0])])

  test_trajectory_loss = loss.TrajectoryLoss(pred=pred_xyz_test, gt=gt_test_dict['xyz'][..., [0, 1, 2]], mask=gt_test_dict['mask'][..., [0, 1, 2]], lengths=gt_test_dict['lengths'])
  test_gravity_loss = loss.GravityLoss(pred=pred_xyz_test, gt=gt_test_dict['xyz'][..., [0, 1, 2]], mask=gt_test_dict['mask'][..., [0, 1, 2]], lengths=gt_test_dict['lengths'])
  # test_depth_loss = loss.DepthLoss(pred=pred_depth_test, gt=gt_test_dict['o_with_f'][..., [0]], lengths=input_test_dict['lengths'], mask=input_test_dict['mask'])
  test_below_ground_loss = loss.BelowGroundPenalize(pred=pred_xyz_test, gt=gt_test_dict['xyz'][..., [0, 1, 2]], mask=gt_test_dict['mask'][..., [0, 1, 2]], lengths=gt_test_dict['lengths'])
  if args.env == 'mocap':
    test_eot_loss = pt.tensor(0.).to(device)
  else:
    test_eot_loss = loss.EndOfTrajectoryLoss(pred=pred_eot_test, gt=gt_test_dict['o_with_f'][..., [1]], mask=input_test_dict['mask'][..., [2]], lengths=input_test_dict['lengths'], startpos=input_test_dict['startpos'][..., [2]], flag='test')
  test_loss = test_trajectory_loss + test_gravity_loss + test_eot_loss

  ####################################
  ############# Evaluation ###########
  ####################################
  # Calculate loss per trajectory
  evaluation_results = evaluateModel(pred=pred_xyz_test, gt=gt_test_dict['xyz'][..., [0, 1, 2]], mask=gt_test_dict['mask'][..., [0, 1, 2]], lengths=gt_test_dict['lengths'], threshold=threshold)
  reconstructed_trajectory = [gt_test_dict['xyz'][..., [0, 1, 2]].detach().cpu().numpy(), pred_xyz_test.detach().cpu().numpy(), input_uv_cumsum_test.detach().cpu().numpy(), pred_depth_cumsum_test.detach().cpu().numpy(), gt_test_dict['lengths'].detach().cpu().numpy()]

  print('Test Loss : {:.3f}'.format(test_loss.item()), end=', ')
  print('Trajectory Loss : {:.3f}'.format(test_trajectory_loss.item()), end=', ')
  print('Gravity Loss : {:.3f}'.format(test_gravity_loss.item()), end=', ')
  print('EndOfTrajectory Loss : {:.3f}'.format(test_eot_loss.item()))

  if vis_flag == True:
    pred_test_dict = {'input':in_test, 'flag':pred_eot_test, 'depth':pred_depth_test, 'xyz':pred_xyz_test}
    utils_inference_func.make_visualize(input_test_dict=input_test_dict, gt_test_dict=gt_test_dict, evaluation_results=evaluation_results, animation_visualize_flag=args.animation_visualize_flag, pred_test_dict=pred_test_dict, visualization_path=visualization_path, args=args)

  return evaluation_results, reconstructed_trajectory

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

def get_model(input_size, output_size, model_arch):
  if model_arch=='bigru_residual_add':
    model_flag = BiGRUResidualAdd(input_size=2, output_size=1)
    model_depth = BiGRUResidualAdd(input_size=3, output_size=1)
  else :
    print("Please input correct model architecture : gru, bigru, lstm, bilstm")
    exit()

  return model_flag, model_depth

def load_checkpoint(model_eot, model_depth):
  if os.path.isfile(args.load_checkpoint):
    print("[#] Found the checkpoint...")
    checkpoint = pt.load(args.load_checkpoint, map_location='cuda:0')
    # Load optimizer, learning rate, decay and scheduler parameters
    model_eot.load_state_dict(checkpoint['model_flag'])
    model_depth.load_state_dict(checkpoint['model_depth'])
    return model_eot, model_depth

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
      summary_evaluation[distance]['accepted_3axis_loss'] += each_batch_eval[distance]['accepted_3axis_loss']
      summary_evaluation[distance]['accepted_3axis_maxdist'] += each_batch_eval[distance]['accepted_3axis_maxdist']

    print("Distance : ", distance)
    print("Mean 3-Axis(X, Y, Z) loss : {}".format(np.mean(summary_evaluation[distance]['loss_3axis'], axis=0)))
    print("SD 3-Axis(X, Y, Z) loss : {}".format(np.std(summary_evaluation[distance]['loss_3axis'], axis=0)))
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
  model_flag, model_depth, model_cfg = utils_func.get_model_depth(model_arch=args.model_arch, features_cols=features_cols, args=args)
  print(model_cfg)
  model_flag = model_flag.to(device)
  model_depth = model_depth.to(device)

  # Load the checkpoint if it's available.
  if args.load_checkpoint is None:
    # Create a model
    print('===>No model checkpoint')
    print('[#] Define the Learning rate, Optimizer, Decay rate and Scheduler...')
  else:
    print('===>Load checkpoint with Optimizer state, Decay and Scheduler state')
    print('[#] Loading ... {}'.format(args.load_checkpoint))
    model_flag, model_depth = load_checkpoint(model_flag, model_depth)

  print('[#]Model Architecture')
  print('####### Model - EOT #######')
  print(model_flag)
  print('####### Model - Depth #######')
  print(model_depth)

  # Test a model iterate over dataloader to get each batch and pass to predict function
  evaluation_results_all = []
  reconstructed_trajectory_all = []
  n_trajectory = 0
  for batch_idx, batch_test in enumerate(trajectory_test_dataloader):
    print("[#]Batch-{}".format(batch_idx))
    # Testing set (Each index in batch_test came from the collate_fn_padd)
    input_test_dict = {'input':batch_test['input'][0].to(device), 'lengths':batch_test['input'][1].to(device), 'mask':batch_test['input'][2].to(device), 'startpos':batch_test['input'][3].to(device)}
    gt_test_dict = {'o_with_f':batch_test['gt'][0].to(device), 'lengths':batch_test['gt'][1].to(device), 'mask':batch_test['gt'][2].to(device), 'startpos':batch_test['gt'][3].to(device), 'xyz':batch_test['gt'][4].to(device)}

      # Call function to test
    evaluation_results, reconstructed_trajectory = predict(input_test_dict=input_test_dict, gt_test_dict=gt_test_dict,
                                 model_flag=model_flag, model_depth=model_depth, vis_flag=args.vis_flag,
                                 threshold=args.threshold, cam_params_dict=cam_params_dict, visualization_path=args.visualization_path)


    reconstructed_trajectory_all.append(reconstructed_trajectory)
    evaluation_results_all.append(evaluation_results)
    n_trajectory += input_test_dict['input'].shape[0]

  summary_evaluation = summary(evaluation_results_all)

  # Save prediction file
  if args.savetofile is not None:
    utils_func.initialize_folder(args.savetofile)
    utils_func.save_reconstructed(eval_metrics=summary_evaluation, trajectory=reconstructed_trajectory_all)
  print("[#] Done")
