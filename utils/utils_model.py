import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
from tqdm import tqdm
sys.path.append(os.path.realpath('../..'))
import utils.transformation as utils_transform
import utils.utils_func as utils_func
import utils.cummulative_depth as utils_cummulative
import utils.loss as utils_loss
# Optimization
from models.Finale.optimization.optimization_refinement import TrajectoryOptimizationRefinement
from models.Finale.optimization.optimization_depth import TrajectoryOptimizationDepth
args = None

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
else:
  device = pt.device('cpu')

def share_args(a):
  global args
  args = a

def fw_pass(model_dict, input_dict, cam_params_dict):
  # Add noise on the fly
  in_f = input_dict['input'][..., [0, 1]].clone()
  if args.noise:
    in_f, missing_dict = utils_func.add_noise(input_trajectory=in_f[..., [0, 1]].clone(), startpos=input_dict['startpos'][..., [0, 1]], lengths=input_dict['lengths'])
  else:
    missing_dict = None
  if 'extrinsic' in args.selected_features:
    extrinsic = utils_func.get_extrinsic_representation(cam_params_dict=cam_params_dict)

  prediction_dict = {}
  pred_eot = pt.tensor([]).to(device)
  features_indexing = 2
  if 'eot' in args.pipeline:
    #############################################
    #################### EOT ####################
    #############################################
    model_flag = model_dict['model_flag']
    pred_eot, (_, _) = model_flag(in_f=in_f, lengths=input_dict['lengths'])
    prediction_dict['model_flag'] = pred_eot
    features_indexing = 3

  if 'latent' in args.pipeline:
    #############################################
    ################### LATENT ##################
    #############################################
    # Will be implemented
    # Considering between 
    # 1. Attention masking
    # 2. EOT flag split
    x = pt.where(input_dict['input'][0][..., 2] == 1.)[0]
    x[-1] = 0
    x = pt.cat((x[[-1]], x[0:-1]))
    for i in range(len(x)):
      print(input_dict['input'][0][x[i]-3:x[i]+3, 3:])
      # print(input_dict['input'][0][x[1]-3:x[1]+3, 3:])
      print(" Cut: ", input_dict['input'][0][x[i], 3:])
      # print(input_dict['input'][0][x[1], 3:])
    exit()

  # Concat the (u_noise, v_noise, pred_eot, other_features(col index 2 if pred_eot is [] else 3)
  if 'refinement' in args.pipeline:
    in_f = pt.cat((in_f, pred_eot), dim=2)
  else:
    in_f = pt.cat((in_f, pred_eot, input_dict['input'][..., features_indexing:]), dim=2)

  if 'depth' in args.pipeline:
    ######################################
    ################ DEPTH ###############
    ######################################
    model_depth = model_dict['model_depth']
    pred_depth, (_, _) = model_depth(in_f=in_f, lengths=input_dict['lengths'])
    prediction_dict['model_depth'] = pred_depth

  return prediction_dict, in_f, missing_dict

def optimize_depth(model_dict, gt_dict, cam_params_dict, optimize, input_dict):
  # Add noise on the fly
  in_f = input_dict['input'][..., [0, 1]].clone()
  if args.noise:
    in_f, missing_dict = utils_func.add_noise(input_trajectory=in_f[..., [0, 1]].clone(), startpos=input_dict['startpos'][..., [0, 1]], lengths=input_dict['lengths'])
  else:
    missing_dict = None
  if 'extrinsic' in args.selected_features:
    extrinsic = utils_func.get_extrinsic_representation(cam_params_dict=cam_params_dict)

  pred_dict = {}
  pred_eot = pt.tensor([]).to(device)
  features_indexing = 2
  if 'eot' in args.pipeline:
    #############################################
    #################### EOT ####################
    #############################################
    model_flag = model_dict['model_flag']
    pred_eot, (_, _) = model_flag(in_f=in_f, lengths=input_dict['lengths'])
    pred_dict['model_flag'] = pred_eot
    features_indexing = 3

  in_f = pt.cat((in_f, pred_eot, input_dict['input'][..., features_indexing:]), dim=2)

  features_indexing = 3
  if 'eot' in args.pipeline:
    features_indexing = 4

  latent_size = model_dict['model_depth'].input_size - features_indexing + 1 # Initial the latent size

  trajectory_optimizer = TrajectoryOptimizationDepth(model_dict=model_dict, uv=in_f, input_dict=input_dict, cam_params_dict=cam_params_dict, latent_size=latent_size, pred_dict=pred_dict, latent_code=args.latent_code)
  latent_optimized = trajectory_optimizer.update_latent()
  trajectory_optimizer.train()
  optimizer = pt.optim.Adam(trajectory_optimizer.parameters(), lr=10)
  lr_scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
  t = tqdm(range(500), desc='Optimizing...', leave=True)
  for i in t:
    optimizer.zero_grad()
    pred_depth_optimized = trajectory_optimizer()
    pred_dict['model_depth'] = pred_depth_optimized

    if args.bi_pred_weight:
      bi_pred_weight = pred_depth_optimized[..., [2]]
    else:
      bi_pred_weight = pt.zeros(pred_depth_optimized[..., [0]].shape)

    pred_depth_cumsum, input_uv_cumsum = utils_cummulative.cummulative_fn(depth=pred_depth_optimized, uv=in_f[..., [0, 1]], depth_teacher=gt_dict['o_with_f'][..., [0]], startpos=input_dict['startpos'], lengths=input_dict['lengths'], eot=pred_eot, cam_params_dict=cam_params_dict, epoch=0, args=args, gt=gt_dict['xyz'][..., [0, 1, 2]], bi_pred_weight=bi_pred_weight)

    # Project the (u, v, depth) to world space
    pred_xyz_optimized = pt.stack([utils_transform.projectToWorldSpace(uv=input_uv_cumsum[i], depth=pred_depth_cumsum[i], cam_params_dict=cam_params_dict, device=device) for i in range(input_uv_cumsum.shape[0])])

    optimization_loss = calculate_optimization_loss(optimized_xyz=pred_xyz_optimized, gt_dict=gt_dict, cam_params_dict=cam_params_dict)
    # for param in trajectory_optimizer.named_parameters()parameters():
      # print(param)
    # for name, param in trajectory_optimizer.named_parameters():
      # if param.requires_grad:
        # print(name, param.data)
    for param_group in optimizer.param_groups:
      lr = param_group['lr']
    t.set_description("Optimizing... (Loss = {}, LR = {})".format(optimization_loss, lr))
    t.refresh()
    if lr < 0.001:
      break
    lr_scheduler.step(optimization_loss)
    optimization_loss.backward(retain_graph=True)
    optimizer.step()

  latent_optimized = trajectory_optimizer.update_latent()
  latent_optimized = pt.cat((latent_optimized[:, [0], :], latent_optimized), dim=1)
  pred_xyz_optimized = pt.cat((pred_xyz_optimized, latent_optimized), dim=2)
  return pred_xyz_optimized, pred_dict, None, input_uv_cumsum, pred_depth_cumsum

def refinement(model_dict, gt_dict, cam_params_dict, pred_xyz, optimize, pred_dict):
  ######################################
  ############# REFINEMENT #############
  ######################################
  features_indexing = 3
  if 'eot' in args.pipeline:
    features_indexing = 4

  if optimize:
    latent_size = model_dict['model_refinement_0'].input_size - features_indexing + 1 # Initial the latent size
    pred_xyz_optimized = optimization_refinement(model_dict=model_dict, pred_xyz=pred_xyz, gt_dict=gt_dict, cam_params_dict=cam_params_dict, latent_size=latent_size, pred_dict=pred_dict)
    return pred_xyz_optimized
  else:
    in_f = pt.cat((pred_xyz, gt_dict['xyz'][..., features_indexing:]), dim=2)
    for idx in range(args.n_refinement):
      model_refinement = model_dict['model_refinement_{}'.format(idx)]
      pred_refinement, (_, _) = model_refinement(in_f=in_f, lengths=gt_dict['lengths'])
      # Fix the 1st and last points Cuz prediction depth with direction or bidirectional is always correct 
      if args.fix_refinement:
        pred_refinement[:, 0, :] = pred_refinement[:, 0, :] * 0.
        for j in range(pred_refinement.shape[0]):
          pred_refinement[j, gt_dict['lengths'][j]-1, :] = pred_refinement[j, gt_dict['lengths'][j]-1, :] * 0.
      pred_xyz = pred_xyz + pred_refinement
    return pred_xyz

def optimization_refinement(model_dict, pred_xyz, gt_dict, cam_params_dict, latent_size, pred_dict):
  trajectory_optimizer = TrajectoryOptimizationRefinement(model_dict=model_dict, pred_xyz=pred_xyz, gt_dict=gt_dict, cam_params_dict=cam_params_dict, latent_size=latent_size, n_refinement=args.n_refinement, pred_dict=pred_dict, latent_code=args.latent_code)
  latent_optimized = trajectory_optimizer.update_latent()
  trajectory_optimizer.train()
  optimizer = pt.optim.Adam(trajectory_optimizer.parameters(), lr=100)
  lr_scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
  t = tqdm(range(500), desc='Optimizing...', leave=True)
  for i in t:
    optimizer.zero_grad()
    pred_xyz_optimized = trajectory_optimizer(pred_xyz)
    optimization_loss = calculate_optimization_loss(optimized_xyz=pred_xyz_optimized, gt_dict=gt_dict, cam_params_dict=cam_params_dict)

    for param_group in optimizer.param_groups:
      lr = param_group['lr']
    t.set_description("Optimizing... (Loss = {}, LR = {})".format(optimization_loss, lr))
    t.refresh()
    if lr < 0.001:
      break
    lr_scheduler.step(optimization_loss)
    optimization_loss.backward(retain_graph=True)
    optimizer.step()

    # with pt.no_grad():
      # for i in range(len(trajectory_optimizer.latent)):
        # trajectory_optimizer.latent[i] = trajectory_optimizer.latent[i].clamp(-0.2, 0.2)
    for name, param in trajectory_optimizer.named_parameters():
      print(name, param)
        # if 'latent' in name:
          # param.copy_(param.clamp(-0.2, 0.2))
          # param = param.clamp(-math.pi, math.pi)
          # print(trajectory_optimizer[name])
          # trajectory_optimizer. = param.clamp(-0.2, 0.2)

  latent_optimized = trajectory_optimizer.update_latent()
  print("FINISH : ", latent_optimized)
  pred_xyz_optimized = pt.cat((pred_xyz_optimized, latent_optimized), dim=2)
  return pred_xyz_optimized

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
  optimization_loss = below_ground_loss + trajectory_loss # + multiview_loss #+ trajectory_loss # 
  return optimization_loss

def train_mode(model_dict):
  for model in model_dict.keys():
    model_dict[model].train()

def eval_mode(model_dict):
  for model in model_dict.keys():
    model_dict[model].eval()

def calculate_loss(pred_xyz, input_dict, gt_dict, cam_params_dict, pred_dict, missing_dict):
  # Calculate loss term
  ######################################
  ############# Trajectory #############
  ######################################
  trajectory_loss = utils_loss.TrajectoryLoss(pred=pred_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
  gravity_loss = utils_loss.GravityLoss(pred=pred_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
  below_ground_loss = utils_loss.BelowGroundPenalize(pred=pred_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])

  ######################################
  ################ Flag ################
  ######################################
  if 'eot' in args.pipeline and args.env == 'unity':
    pred_eot = pred_dict['model_flag']
    eot_loss = utils_loss.EndOfTrajectoryLoss(pred=pred_eot, gt=gt_dict['o_with_f'][..., [1]], mask=input_dict['mask'][..., [2]], lengths=input_dict['lengths'], startpos=input_dict['startpos'][..., [2]], flag='Train')
  else:
    eot_loss = pt.tensor(0.)

  ######################################
  ############ Reprojection ############
  ######################################

  if len(args.multiview_loss) > 0:
    multiview_loss = utils_loss.MultiviewReprojectionLoss(pred=pred_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1]], lengths=gt_dict['lengths'], cam_params_dict=cam_params_dict)
  else:
    multiview_loss = pt.tensor(0.)


  ######################################
  ########### Interpolation ############
  ######################################
  if args.missing != None:
    uv_pred = pt.cat((pt.unsqueeze(input_dict['input'][:, [0], [0, 1]], dim=1), pred_dict['model_depth'][:, :-1, [2, 3]]), dim=1)
    interpolation_loss = utils_loss.InterpolationLoss(uv_gt=input_dict['input'][..., [0, 1]], uv_pred=uv_pred, mask=input_dict['mask'][..., [0, 1]], lengths=input_dict['lengths'])
  else:
    interpolation_loss = pt.tensor(0.)

  ######################################
  ############### Depth ################
  ######################################
  if args.bi_pred_ramp:
    depth_loss_fw = utils_loss.DepthLoss(pred=pred_dict['model_depth'][..., [0]], gt=gt_dict['o_with_f'][..., [0]], lengths=input_dict['lengths'], mask=input_dict['mask'][..., [0]])
    depth_loss_bw = utils_loss.DepthLoss(pred=pred_dict['model_depth'][..., [1]], gt=-gt_dict['o_with_f'][..., [0]], lengths=input_dict['lengths'], mask=input_dict['mask'][..., [0]])
    depth_loss = depth_loss_fw + depth_loss_bw

  # Sum up all loss 
  loss = trajectory_loss + eot_loss*10 + gravity_loss + below_ground_loss + multiview_loss + interpolation_loss*10 + depth_loss*100
  loss_dict = {"Trajectory Loss":trajectory_loss.item(), "EndOfTrajectory Loss":eot_loss.item()*10, "Gravity Loss":gravity_loss.item(), "BelowGroundPenalize Loss":below_ground_loss.item(), "MultiviewReprojection Loss":multiview_loss.item(), "Interpolation Loss":interpolation_loss.item()*10, "Depth Loss":depth_loss.item()*100}

  return loss_dict, loss
