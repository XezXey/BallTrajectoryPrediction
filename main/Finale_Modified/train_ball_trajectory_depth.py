from __future__ import print_function
# Import libs
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
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
import utils.utils_model as utils_model
# Loss
import utils.loss as utils_loss

# Argumentparser for input
parser = argparse.ArgumentParser(description='Predict the 3D projectile')
parser.add_argument('--dataset_train_path', dest='dataset_train_path', type=str, help='Path to training set', required=True)
parser.add_argument('--dataset_val_path', dest='dataset_val_path', type=str, help='Path to validation set', required=True)
parser.add_argument('--batch_size', dest='batch_size', type=int, help='Size of batch', default=50)
parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, help='Type of trajectory(Rolling, Projectile, MagnusProjectile)', default='Projectile')
parser.add_argument('--no_visualize', dest='vis_flag', help='No Visualize the trajectory', action='store_false')
parser.add_argument('--visualize', dest='vis_flag', help='Visualize the trajectory', action='store_true')
parser.add_argument('--save_checkpoint', dest='save_checkpoint', type=str, help='Path to save a model checkpoint', required=True)
parser.add_argument('--load_checkpoint', dest='load_checkpoint', type=str, help='Path to load a trained model checkpoint', default=None)
parser.add_argument('--visualization_path', dest='visualization_path', type=str, help='Path to visualization directory', default='./visualize_html/')
parser.add_argument('--cam_params_file', dest='cam_params_file', type=str, help='Path to camera parameters file(Intrinsic/Extrinsic)')
parser.add_argument('--wandb_name', dest='wandb_name', type=str, help='WanDB session name', default=None, required=True)
parser.add_argument('--wandb_notes', dest='wandb_notes', type=str, help='WanDB notes', default=None)
parser.add_argument('--wandb_tags', dest='wandb_tags', type=str, help='WanDB tags name', default=None, required=True)
parser.add_argument('--cuda_device_num', dest='cuda_device_num', type=int, help='Provide cuda device number', default=0)
parser.add_argument('--clip', dest='clip', type=float, help='Clipping gradients value', required=True)
parser.add_argument('--model_arch', dest='model_arch', help='Input the model architecture(lstm, bilstm, gru, bigru)', nargs='+', default=[])
parser.add_argument('--noise', dest='noise', help='Noise on the fly', action='store_true')
parser.add_argument('--no_noise', dest='noise', help='Noise on the fly', action='store_false')
parser.add_argument('--noise_sd', dest='noise_sd', help='Std. of noise', type=float, default=None)
parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('--decay_gamma', help='Gamma (Decay rate)', type=float, default=0.7)
parser.add_argument('--decay_cycle', help='Decay cycle', type=int, default=50)
parser.add_argument('--teacherforcing_depth', help='Use a teacher forcing training scheme for depth displacement estimation', action='store_true', default=False)
parser.add_argument('--teacherforcing_mixed', help='Use a teacher forcing training scheme for depth displacement estimation on some part of training set', action='store_true', default=False)
parser.add_argument('--wandb_dir', help='Path to WanDB directory', type=str, default='./')
parser.add_argument('--start_decumulate', help='Epoch to start training with decumulate of an error', type=int, default=0)
parser.add_argument('--decumulate', help='Decumulate the depth by ray casting', action='store_true', default=False)
parser.add_argument('--selected_features', dest='selected_features', help='Specify the selected features columns(eot, og, ', nargs='+', default=[])
parser.add_argument('--bi_pred_avg', help='Bidirectional prediction', action='store_true', default=False)
parser.add_argument('--bi_pred_weight', help='Bidirectional prediction with weight', action='store_true', default=False)
parser.add_argument('--bi_pred_ramp', help='Bidirectional prediction with ramp weight', action='store_true', default=False)
parser.add_argument('--bw_pred', help='Backward prediction', action='store_true', default=False)
parser.add_argument('--trainable_init', help='Trainable initial state', action='store_true', default=False)
parser.add_argument('--env', dest='env', help='Environment', default='unity')
parser.add_argument('--bidirectional', dest='bidirectional', help='Define use of bidirectional', nargs='+', default=[])
parser.add_argument('--directional', dest='bidirectional', help='Define use of bidirectional', action='store_false')
parser.add_argument('--multiview_loss', dest='multiview_loss', help='Use Multiview loss', nargs='+', default=[])
parser.add_argument('--pipeline', dest='pipeline', help='Pipeline', nargs='+', default=[])
parser.add_argument('--n_refinement', dest='n_refinement', help='Refinement Iterations', type=int, default=1)
parser.add_argument('--fix_refinement', dest='fix_refinement', help='Fix Refinement for 1st and last points', action='store_true', default=False)
parser.add_argument('--optimize', dest='optimize', help='Flag to optimze(This will work when train with latent', default=None)
parser.add_argument('--refine', dest='refine', help='I/O for refinement network', default='position')
parser.add_argument('--recon', dest='recon', help='UV selection', default='ideal_uv')
parser.add_argument('--autoregressive', dest='autoregressive', help='Doing auto_regression for interpolation', action='store_true', default=False)
parser.add_argument('--in_refine', dest='in_refine', help='Input for refinement network', default='xyz')
parser.add_argument('--out_refine', dest='out_refine', help='Output for refinement network', default='xyz')
parser.add_argument('--annealing', dest='annealing', help='Apply annealing', action='store_true', default=False)
parser.add_argument('--annealing_cycle', dest='annealing_cycle', type=int, help='Apply annealing every n epochs', default=5)
parser.add_argument('--annealing_gamma', dest='annealing_gamma', type=float, help='Apply annealing every n epochs', default=0.95)
parser.add_argument('--latent_transf', dest='latent_transf', type=str, help='Extra latent manipulation method', default=None)
args = parser.parse_args()


# Share args to every modules
utils_func.share_args(args)
utils_transform.share_args(args)
utils_cummulative.share_args(args)
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

# Init wandb
if args.wandb_notes is None:
  args.wandb_notes = args.wandb_name
wandb.init(project="ball-trajectory-estimation", name=args.wandb_name, tags=args.wandb_tags, notes=args.wandb_notes, dir=args.wandb_dir)
# Get selected features to input into a network
features = ['x', 'y', 'z', 'u', 'v', 'd', 'eot', 'og', 'rad', 'f_sin', 'f_cos', 'fx', 'fy', 'fz', 'fx_norm', 'fy_norm', 'fz_norm', 'g']
x, y, z, u, v, d, eot, og, rad, f_sin, f_cos, fx, fy, fz, fx_norm, fy_norm, fz_norm, g = range(len(features))
input_col, input_startpos_col, gt_col, gt_startpos_col, gt_xyz_col, features_cols = utils_func.get_selected_cols(args=args, pred='depth')

def train(input_train_dict, gt_train_dict, input_val_dict, gt_val_dict, model_dict, epoch, vis_signal, optimizer, cam_params_dict, annealing_weight, vis_flag=True, visualization_path='./visualize_html/'):
  # Training RNN/LSTM model
  # Run over each example
  # Train a model

  ####################################
  ############# Training #############
  ####################################
  utils_model.train_mode(model_dict=model_dict)

  pred_dict_train, in_train, missing_dict_train = utils_model.fw_pass(model_dict, input_dict=input_train_dict, cam_params_dict=cam_params_dict)
  pred_depth_train, pred_flag_train, input_flag_train = utils_func.get_pipeline_var(pred_dict=pred_dict_train, input_dict=input_train_dict)

  if args.bi_pred_weight:
    bi_pred_weight_train = pred_depth_train[..., [2]]
  else:
    bi_pred_weight_train = pt.zeros(pred_depth_train[..., [0]].shape)

  uv_train = utils_func.select_uv_recon(input_dict=input_train_dict, pred_dict=pred_dict_train, in_f_noisy=in_train)

  pred_depth_cumsum_train, input_uv_cumsum_train = utils_cummulative.cummulative_fn(depth=pred_depth_train, uv=uv_train, depth_teacher=gt_train_dict['o_with_f'][..., [0]], startpos=input_train_dict['startpos'], lengths=input_train_dict['lengths'], eot=input_flag_train, cam_params_dict=cam_params_dict, epoch=epoch, args=args, gt=gt_train_dict['xyz'][..., [0, 1, 2]], bi_pred_weight=bi_pred_weight_train)

  # Project the (u, v, depth) to world space
  pred_xyz_train = pt.stack([utils_transform.projectToWorldSpace(uv=input_uv_cumsum_train[i], depth=pred_depth_cumsum_train[i], cam_params_dict=cam_params_dict, device=device) for i in range(input_uv_cumsum_train.shape[0])])

  if 'refinement' in args.pipeline:
    pred_xyz_refined_train = utils_model.refinement(model_dict=model_dict, gt_dict=gt_train_dict, cam_params_dict=cam_params_dict, pred_xyz=pred_xyz_train, optimize=args.optimize, pred_dict=pred_dict_train)
  else: pred_xyz_refined_train = None

  optimizer.zero_grad() # Clear existing gradients from previous epoch
  train_loss_dict, train_loss = utils_model.calculate_loss(pred_xyz=pred_xyz_train, pred_xyz_refined=pred_xyz_refined_train, input_dict=input_train_dict, gt_dict=gt_train_dict, cam_params_dict=cam_params_dict, pred_dict=pred_dict_train, missing_dict=missing_dict_train, annealing_weight=annealing_weight) # Calculate the loss

  train_loss.backward()

  # for model in model_dict.keys():
    # print(model)
    # if 'refinement' in model:
      # for name, param in model_dict[model].named_parameters():
        # print(name, param)

  for model in model_dict:
    pt.nn.utils.clip_grad_norm_(model_dict[model].parameters(), args.clip)
  optimizer.step()

  ####################################
  ############# Evaluate #############
  ####################################
  # Evaluating mode
  utils_model.eval_mode(model_dict=model_dict)

  pred_dict_val, in_val, missing_dict_val = utils_model.fw_pass(model_dict, input_dict=input_val_dict, cam_params_dict=cam_params_dict)
  pred_depth_val, pred_flag_val, input_flag_val = utils_func.get_pipeline_var(pred_dict=pred_dict_val, input_dict=input_val_dict)

  if args.bi_pred_weight:
    bi_pred_weight_val = pred_depth_val[..., [2]]
  else:
    bi_pred_weight_val = pt.zeros(pred_depth_val[..., [0]].shape)

  uv_val = utils_func.select_uv_recon(input_dict=input_val_dict, pred_dict=pred_dict_val, in_f_noisy=in_val)

  pred_depth_cumsum_val, input_uv_cumsum_val = utils_cummulative.cummulative_fn(depth=pred_depth_val, uv=uv_val, depth_teacher=gt_val_dict['o_with_f'][..., [0]], startpos=input_val_dict['startpos'], lengths=input_val_dict['lengths'], eot=input_flag_val, cam_params_dict=cam_params_dict, epoch=epoch, args=args, gt=gt_val_dict['xyz'][..., [0, 1, 2]], bi_pred_weight=bi_pred_weight_val)

  # Project the (u, v, depth) to world space
  pred_xyz_val = pt.stack([utils_transform.projectToWorldSpace(uv=input_uv_cumsum_val[i], depth=pred_depth_cumsum_val[i], cam_params_dict=cam_params_dict, device=device) for i in range(input_uv_cumsum_val.shape[0])])

  if 'refinement' in args.pipeline:
    pred_xyz_refined_val = utils_model.refinement(model_dict=model_dict, gt_dict=gt_val_dict, cam_params_dict=cam_params_dict, pred_xyz=pred_xyz_val, optimize=args.optimize, pred_dict=pred_dict_val)
  else: pred_xyz_refined_val = None


  optimizer.zero_grad() # Clear existing gradients from previous epoch
  val_loss_dict, val_loss = utils_model.calculate_loss(pred_xyz=pred_xyz_val, pred_xyz_refined=pred_xyz_refined_val, input_dict=input_val_dict, gt_dict=gt_val_dict, cam_params_dict=cam_params_dict, pred_dict=pred_dict_val, missing_dict=missing_dict_val, annealing_weight=annealing_weight) # Calculate the loss

  utils_func.print_loss(loss_list=[train_loss_dict, train_loss], name='Training')
  utils_func.print_loss(loss_list=[val_loss_dict, val_loss], name='Validating')
  wandb.log({'Train Loss':train_loss.item(), 'Validation Loss':val_loss.item()})

  if vis_flag == True and vis_signal == True:
    if 'uv' not in args.pipeline:
      missing_dict_train = {'mask':None, 'idx':None}
      missing_dict_val = {'mask':None, 'idx':None}

    pred_xyz_train_finale = pred_xyz_train if 'refinement' not in args.pipeline else pred_xyz_refined_train
    pred_xyz_val_finale = pred_xyz_val if 'refinement' not in args.pipeline else pred_xyz_refined_val
    pred_train_dict = {'input':in_train, 'flag':pred_flag_train, 'depth':pred_depth_train, 'xyz':pred_xyz_train, 'xyz_refined':pred_xyz_refined_train, 'finale_xyz':pred_xyz_train_finale, 'missing_mask':missing_dict_train['mask'], 'missing_idx':missing_dict_train['idx'], 'pred_uv':uv_train}
    pred_val_dict = {'input':in_val, 'flag':pred_flag_val, 'depth':pred_depth_val, 'xyz':pred_xyz_val, 'xyz_refined':pred_xyz_refined_train, 'finale_xyz':pred_xyz_val_finale, 'missing_mask':missing_dict_val['mask'], 'missing_idx':missing_dict_val['idx'], 'pred_uv':uv_val}
    utils_func.make_visualize(input_train_dict=input_train_dict, gt_train_dict=gt_train_dict, input_val_dict=input_val_dict, gt_val_dict=gt_val_dict, pred_train_dict=pred_train_dict, pred_val_dict=pred_val_dict, visualization_path=visualization_path, pred='depth', cam_params_dict=cam_params_dict)


  return train_loss.item(), val_loss.item(), model_dict

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


if __name__ == '__main__':
  print('[#]Training : Trajectory Estimation')

  # Initialize folder
  utils_func.initialize_folder(args.visualization_path)
  save_checkpoint = '{}/{}/'.format(args.save_checkpoint + args.wandb_tags.replace('/', '_'), args.wandb_name)
  utils_func.initialize_folder(save_checkpoint)

  # Load camera parameters : ProjectionMatrix and WorldToCameraMatrix
  cam_params_dict = utils_transform.get_cam_params_dict(args.cam_params_file, device)

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
  min_val_loss = 2e10
  annealing_step = 0
  annealing_weight = 1.0
  model_dict, model_cfg = utils_func.get_model_depth(model_arch=args.model_arch, features_cols=features_cols, args=args)
  print(model_dict)
  print(model_cfg)
  model_dict = {model:model_dict[model].to(device) for model in model_dict.keys()}

  # Define optimizer, learning rate, decay and scheduler parameters
  # for model
  params = []
  for model in model_dict.keys():
    if 'refinement' not in model:
      print(model_dict[model].parameters())
      # print("PARAM TO OPTIM : ", model)
    params += list(model_dict[model].parameters())
  optimizer = pt.optim.Adam(params, lr=args.lr)
  lr_scheduler = pt.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.decay_gamma)
  start_epoch = 1

  # Load the checkpoint if it's available.
  if args.load_checkpoint is None:
    # Create a model
    print('===>No model checkpoint')
    print('[#] Define the Learning rate, Optimizer, Decay rate and Scheduler...')
  else:
    print('===>Load checkpoint with Optimizer state, Decay and Scheduler state')
    print('[#] Loading ... {}'.format(args.load_checkpoint))
    model_dict, optimizer, start_epoch, lr_scheduler, min_val_loss, annealing_scheduler = utils_func.load_checkpoint_train(model_dict, optimizer, lr_scheduler)
    annealing_step = annealing_scheduler['step']
    annealing_weight = annealing_scheduler['weight']

  print('[#]Model Architecture')
  for model in model_cfg.keys():
    print('####### Model - {} #######'.format(model))
    print(model_dict[model])
    # Log metrics with wandb
    wandb.watch(model_dict[model])

  # Training settings
  n_epochs = 100000
  for epoch in range(start_epoch, n_epochs+1):
    accumulate_train_loss = []
    accumulate_val_loss = []
    # Fetch the Validation set (Get each batch for each training epochs)
    try:
      batch_val = next(trajectory_val_iterloader)
    except StopIteration:
      trajectory_val_iterloader = iter(trajectory_val_dataloader)
      batch_val = next(trajectory_val_iterloader)

    input_val_dict = {'input':batch_val['input'][0].to(device), 'lengths':batch_val['input'][1].to(device), 'mask':batch_val['input'][2].to(device), 'startpos':batch_val['input'][3].to(device)}
    gt_val_dict = {'o_with_f':batch_val['gt'][0].to(device), 'lengths':batch_val['gt'][1].to(device), 'mask':batch_val['gt'][2].to(device), 'startpos':batch_val['gt'][3].to(device), 'xyz':batch_val['gt'][4].to(device)}

    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>[Epoch : {}/{}]<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(epoch, n_epochs))

    # Log the learning rate
    for param_group in optimizer.param_groups:
      print("[#]Learning rate (Depth & EOT) : ", param_group['lr'])
      wandb.log({'Learning Rate':param_group['lr']})

    # Visualize signal to make a plot and save to wandb every epoch is done.
    vis_signal = True if epoch % 20 == 0 else False

    # Training a model iterate over dataloader to get each batch and pass to train function
    # '''
    for batch_idx, batch_train in enumerate(trajectory_train_dataloader):
      print('===> [Minibatch {}/{}].........'.format(batch_idx+1, len(trajectory_train_dataloader)), end='\n')
      # Training set (Each index in batch_train came from the collate_fn_padd)
      input_train_dict = {'input':batch_train['input'][0].to(device), 'lengths':batch_train['input'][1].to(device), 'mask':batch_train['input'][2].to(device), 'startpos':batch_train['input'][3].to(device)}
      gt_train_dict = {'o_with_f':batch_train['gt'][0].to(device), 'lengths':batch_train['gt'][1].to(device), 'mask':batch_train['gt'][2].to(device), 'startpos':batch_train['gt'][3].to(device), 'xyz':batch_train['gt'][4].to(device)}

      # Call function to train
      train_loss, val_loss, model_dict = train(input_train_dict=input_train_dict, gt_train_dict=gt_train_dict, input_val_dict=input_val_dict, gt_val_dict=gt_val_dict, annealing_weight=annealing_weight,
                                                            model_dict=model_dict, vis_flag=args.vis_flag,
                                                            optimizer=optimizer, epoch=epoch, vis_signal=vis_signal,
                                                            cam_params_dict=cam_params_dict, visualization_path=args.visualization_path)

      accumulate_val_loss.append(val_loss)
      accumulate_train_loss.append(train_loss)
      vis_signal = False
    # '''

    # Get the average loss for each epoch over entire dataset
    val_loss_per_epoch = np.mean(accumulate_val_loss)
    train_loss_per_epoch = np.mean(accumulate_train_loss)
    # Log the each epoch loss
    wandb.log({'Epoch Train Loss':train_loss_per_epoch, 'Epoch Validation Loss':val_loss_per_epoch})

    # Decrease learning rate every n_epochs % decay_cycle batch
    if epoch % args.decay_cycle == 0:
      lr_scheduler.step()
      for param_group in optimizer.param_groups:
        print("[#]Stepping Learning rate to ", param_group['lr'])

    # Decrease learning rate every n_epochs % annealing_cycle batch
    if epoch % args.annealing_cycle == 0:
      annealing_weight = annealing_weight * args.annealing_gamma ** annealing_step
      annealing_step += 1
      print("[#]Stepping annealing weight to ", annealing_weight)


    # Save the model checkpoint every finished the epochs
    print('[#]Finish Epoch : {}/{}.........Train loss : {:.3f}, Val loss : {:.3f}'.format(epoch, n_epochs, train_loss_per_epoch, val_loss_per_epoch))
    if min_val_loss > val_loss_per_epoch:
      # Save model checkpoint
      save_checkpoint_best = '{}/{}_best.pth'.format(save_checkpoint, args.wandb_name)
      print('[+++]Saving the best model checkpoint : Prev loss {:.3f} > Curr loss {:.3f}'.format(min_val_loss, val_loss_per_epoch))
      print('[+++]Saving the best model checkpoint to : ', save_checkpoint_best)
      min_val_loss = val_loss_per_epoch
      annealing_scheduler = {'step':annealing_step, 'weight':annealing_weight}
      # Save to directory
      checkpoint = {'epoch':epoch+1, 'optimizer':optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict(), 'min_val_loss':min_val_loss, 'model_cfg':model_cfg, 'annealing_scheduler':annealing_scheduler}
      for model in model_cfg:
        checkpoint[model] = model_dict[model].state_dict()
      pt.save(checkpoint, save_checkpoint_best)
      pt.save(checkpoint, os.path.join(wandb.run.dir, 'checkpoint_best.pth'))

    else:
      print('[#]Not saving the best model checkpoint : Val loss {:.3f} not improved from {:.3f}'.format(val_loss_per_epoch, min_val_loss))


    if epoch % 20 == 0:
      # Save the lastest checkpoint for continue training every 10 epoch
      save_checkpoint_lastest = '{}/{}_lastest.pth'.format(save_checkpoint, args.wandb_name)
      print('[#]Saving the lastest checkpoint to : ', save_checkpoint_lastest)
      checkpoint = {'epoch':epoch+1, 'optimizer':optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict(), 'min_val_loss':min_val_loss, 'model_cfg':model_cfg, 'annealing_scheduler':annealing_scheduler}
      for model in model_cfg:
        checkpoint[model] = model_dict[model].state_dict()
      pt.save(checkpoint, save_checkpoint_lastest)
      pt.save(checkpoint, os.path.join(wandb.run.dir, 'checkpoint_lastest.pth'))

  print("[#] Done")
