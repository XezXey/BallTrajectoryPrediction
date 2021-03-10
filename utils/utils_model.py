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
# from models.Finale.optimization.optimization_refinement import TrajectoryOptimizationRefinement
# from models.Finale.optimization.optimization_depth import TrajectoryOptimizationDepth
from models.Finale.optimization.optimization import TrajectoryOptimizationRefinement, TrajectoryOptimizationDepth
args = None

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
else:
  device = pt.device('cpu')

def share_args(a):
  global args
  args = a

def uv_forward(model_dict, input_dict, in_f, prediction_dict):
  # FORWARD DIRECTION
  model_uv_fw = model_dict['model_uv_fw']
  if args.pred_uv_space == 'pixel':
    lengths_fw = input_dict['lengths']+1
    in_f_uv_fw = pt.cumsum(pt.cat((input_dict['startpos'][..., [0, 1]], in_f), dim=1), dim=1)

  elif args.pred_uv_space == 'cumsum' or args.pred_uv_space == 'delta' or args.pred_uv_space == 'consec':
    lengths_fw = input_dict['lengths']
    in_f_uv_fw = in_f

  pred_uv_fw, (_, _) = model_uv_fw(in_f=in_f_uv_fw, lengths=lengths_fw)
  pred_uv_fw = pt.cat((pt.unsqueeze(in_f_uv_fw[:, [0], [0, 1]], dim=1), pred_uv_fw[:, :-1, [0, 1]]), dim=1)

  # BACKWARD DIRECTION
  model_uv_bw = model_dict['model_uv_bw']
  if args.pred_uv_space == 'pixel':
    lengths_bw = input_dict['lengths']+1
    in_f_uv_bw = pt.cumsum(pt.cat((input_dict['startpos'][..., [0, 1]], in_f), dim=1), dim=1)
    in_f_uv_bw = utils_func.reverse_masked_seq(seq=in_f_uv_bw[..., [0, 1]].clone(), lengths=lengths_bw)

  elif args.pred_uv_space == 'cumsum' or args.pred_uv_space == 'delta' or args.pred_uv_space == 'consec':
    lengths_bw = input_dict['lengths']
    in_f_uv_bw = utils_func.reverse_masked_seq(seq=in_f[..., [0, 1]].clone(), lengths=lengths_bw)

  pred_uv_bw, (_, _) = model_uv_bw(in_f=in_f_uv_bw, lengths=lengths_bw)
  pred_uv_bw = pt.cat((pt.unsqueeze(in_f_uv_bw[:, [0], [0, 1]], dim=1), pred_uv_bw[:, :-1, [0, 1]]), dim=1)

  # Use the interpolate_missing to combine 2 direction into one displacement
  prediction_dict['model_uv_fw'] = pred_uv_fw
  prediction_dict['model_uv_bw'] = pred_uv_bw

  pred_uv = utils_func.combine_uv_bidirection(pred_dict=prediction_dict, input_dict=input_dict, mode=args.pred_uv_space, missing_annotation=None)
  prediction_dict['model_uv'] = pred_uv

  return prediction_dict


def uv_auto_regressive_forward(model_dict, input_dict, in_f, prediction_dict):
  '''
  Missing convention
  1 = missing
  0 = not missing
  '''
  model_uv_fw = model_dict['model_uv_fw']
  model_uv_bw = model_dict['model_uv_bw']

  batch_size = in_f.shape[0]
  pred_uv_fw = [[] for i in range(batch_size)]
  pred_uv_bw = [[] for i in range(batch_size)]

  # Pre-inverse the backward sequence
  if args.pred_uv_space == 'pixel':
    lengths = input_dict['lengths'] + 1
    in_f_uv_fw = pt.cumsum(pt.cat((input_dict['startpos'][..., [0, 1]], in_f), dim=1), dim=1)
    in_f_uv_bw = pt.cumsum(pt.cat((input_dict['startpos'][..., [0, 1]], in_f), dim=1), dim=1)
    in_f_uv_bw = utils_func.reverse_masked_seq(seq=in_f_uv_bw[..., [0, 1]].clone(), lengths=lengths)

  elif args.pred_uv_space == 'cumsum' or args.pred_uv_space == 'delta' or args.pred_uv_space == 'consec' or args.pred_uv_space == 'consec_ar':
    lengths = input_dict['lengths']
    in_f_uv_bw = utils_func.reverse_masked_seq(seq=in_f[..., [0, 1]].clone(), lengths=lengths)
    in_f_uv_fw = in_f

  # Missing annotation
  # This is mock-up version (Real data will loaded from tracking file)
  if not args.load_missing:
    missing_uv = pt.randint(low=0, high=2, size=(in_f_uv_fw.shape[0], in_f_uv_fw.shape[1]+1, 1)).cuda()
    missing_uv[:, :2, 0] *= 0
    missing_uv[:, -2:, 0] *= 0
    input_dict['in_f_missing_uv'] = missing_uv
    missing_duv = missing_uv[:, :-1, :] | missing_uv[:, 1:, :]
    input_dict['in_f_missing_duv'] = missing_duv
    missing_annotation = missing_duv.cuda()
  else:
    missing_uv = input_dict['missing_mask'].int().cuda()
    input_dict['in_f_missing_uv'] = missing_uv
    missing_duv = missing_uv[:, :-1, :] | missing_uv[:, 1:, :]
    input_dict['in_f_missing_duv'] = missing_duv
    missing_annotation = missing_duv.cuda()

  for batch_idx in range(batch_size):    # Batch dimension
    # Reset initial state
    model_uv_fw.set_init()
    model_uv_bw.set_init()

    # Duv missing
    missing_fw = missing_annotation[batch_idx, ...]
    missing_bw = pt.flip(missing_annotation[batch_idx, ...], dims=[0])

    for i in tqdm(range(lengths[batch_idx]-1)):   # Sequence length => #n times to predict
      ###########################################################################################
      #################################### FORWARD DIRECTION ####################################
      ###########################################################################################
      # Input selection
      if missing_fw[i] == 1:
        # Never happend when i=0
        # The point is missing => Use the previous point (t-1)
        in_f_uv_fw_temp = pred_uv_fw[batch_idx][i-1]
      else:
        # The tracking is not missing => Use tracking
        in_f_uv_fw_temp = pt.unsqueeze(in_f_uv_fw[[batch_idx], [i], ...], dim=0)

      pred_uv_fw_temp, (_, _) = model_uv_fw(in_f=in_f_uv_fw_temp, lengths=None)
      print("INPUT : ", in_f_uv_fw_temp)
      print("PREDICTION : ", pred_uv_fw_temp)
      print("="*100)

      # Output selection
      if missing_fw[i+1] == 1:
        # The point is missing => Use the previous point (t-1)
        pred_uv_fw[batch_idx].append(pred_uv_fw_temp)
      else:
        # The tracking is not missing => Use tracking (t+1)
        in_f_uv_fw_t_future = pt.unsqueeze(in_f_uv_fw[[batch_idx], [i+1], ...], dim=0)
        pred_uv_fw[batch_idx].append(in_f_uv_fw_t_future)

      ###########################################################################################
      ################################### BACKWARD DIRECTION ####################################
      ###########################################################################################
      if missing_bw[i] == 1:
        # Never happend when i=0
        # The point is missing => Use the previous point (t-1)
        in_f_uv_bw_temp = pred_uv_bw[batch_idx][i-1]
      else:
        # The tracking is not missing => Use tracking
        in_f_uv_bw_temp = pt.unsqueeze(in_f_uv_bw[[batch_idx], [i], ...], dim=0)

      pred_uv_bw_temp, (_, _) = model_uv_bw(in_f=in_f_uv_bw_temp, lengths=None)

      if missing_bw[i+1] == 1:
        # The point is missing => Use the previous point (t-1)
        pred_uv_bw[batch_idx].append(pred_uv_bw_temp)
      else:
        # The tracking is not missing => Use tracking
        in_f_uv_bw_t_future = pt.unsqueeze(in_f_uv_bw[[batch_idx], [i+1], ...], dim=0)
        pred_uv_bw[batch_idx].append(in_f_uv_bw_t_future)

    # One long sequence
    pred_uv_fw[batch_idx] = pt.cat(pred_uv_fw[batch_idx], dim=1)
    pred_uv_bw[batch_idx] = pt.cat(pred_uv_bw[batch_idx], dim=1)

  # Padding first
  # If loop using sequence length ===> Padding
  pred_uv_fw = pt.nn.utils.rnn.pad_sequence([pred_uv_fw[i][0] for i in range(batch_size)], batch_first=True)
  pred_uv_bw = pt.nn.utils.rnn.pad_sequence([pred_uv_bw[i][0] for i in range(batch_size)], batch_first=True)

  # du0, dv0 concat (Note that du0, dv0 is always exists)
  if args.pred_uv_space == 'pixel':
    pred_uv_fw = pt.cat((pt.unsqueeze(in_f_uv_fw[:, [0], [0, 1]], dim=1), pred_uv_fw[:, :, [0, 1]]), dim=1)
    pred_uv_bw = pt.cat((pt.unsqueeze(in_f_uv_bw[:, [0], [0, 1]], dim=1), pred_uv_bw[:, :, [0, 1]]), dim=1)
  elif args.pred_uv_space == 'delta' or args.pred_uv_space == 'cumsum' or args.pred_uv_space == 'consec' or args.pred_uv_space == 'consec_ar':
    pred_uv_fw = pt.cat((pt.unsqueeze(in_f_uv_fw[:, [0], [0, 1]], dim=1), pred_uv_fw[:, :, [0, 1]]), dim=1)
    pred_uv_bw = pt.cat((pt.unsqueeze(in_f_uv_bw[:, [0], [0, 1]], dim=1), pred_uv_bw[:, :, [0, 1]]), dim=1)

  # Use the interpolate_missing to combine 2 direction into one displacement
  prediction_dict['model_uv_fw'] = pred_uv_fw
  prediction_dict['model_uv_bw'] = pred_uv_bw

  pred_uv = utils_func.combine_uv_bidirection(pred_dict=prediction_dict, input_dict=input_dict, mode=args.pred_uv_space, missing_annotation=[missing_fw, missing_bw, missing_uv])
  prediction_dict['model_uv'] = pred_uv

  return prediction_dict

def fw_pass(model_dict, input_dict, cam_params_dict):
  # Add noise on the fly
  in_f = input_dict['input'][..., [0, 1]].clone()
  if args.noise:
    in_f, missing_dict = utils_func.add_noise(input_trajectory=in_f[..., [0, 1]].clone(), startpos=input_dict['startpos'][..., [0, 1]], lengths=input_dict['lengths'])
  else:
    _, missing_dict = utils_func.add_noise(input_trajectory=in_f[..., [0, 1]].clone(), startpos=input_dict['startpos'][..., [0, 1]], lengths=input_dict['lengths'])

  prediction_dict = {}
  pred_eot = pt.tensor([]).to(device)
  features_indexing = 2

  if 'uv' in args.pipeline:
    #############################################
    #################### UV #####################
    #############################################
    '''
    Make a prediction then
    - concat the first timestep to the prediction
    - ignore the last timestep prediction
    Because we make a prediction of t+1
    '''
    if args.autoregressive:
      # Auto regression forward here.
      prediction_dict = uv_auto_regressive_forward(model_dict=model_dict, input_dict=input_dict, in_f=in_f, prediction_dict=prediction_dict)
    else:
      # Regular forward pass
      prediction_dict = uv_forward(model_dict=model_dict, input_dict=input_dict, in_f=in_f, prediction_dict=prediction_dict)

    pred_uv = prediction_dict['model_uv']

  if 'eot' in args.pipeline:
    #############################################
    #################### EOT ####################
    #############################################
    if 'uv' in args.pipeline:
      in_f_eot = pred_uv
    else:
      in_f_eot = in_f
    model_flag = model_dict['model_flag']
    pred_eot, (_, _) = model_flag(in_f=in_f_eot, lengths=input_dict['lengths'])
    prediction_dict['model_flag'] = pred_eot
    features_indexing = 3


  if 'depth' in args.pipeline:
    ######################################
    ################ DEPTH ###############
    ######################################
    # Input selection from autoregressive network or input directly
    if 'uv' in args.pipeline:
      in_f_depth = pred_uv
    else:
      in_f_depth = in_f

    # Concat the (u_noise, v_noise, pred_eot, other_features(col index 2 if pred_eot is [] else 3)
    if args.optimize == 'depth' or args.optimize == 'both':
      in_f_depth = pt.cat((in_f_depth, pred_eot, input_dict['input'][..., features_indexing:]), dim=2)
    else:
      in_f_depth = pt.cat((in_f_depth, pred_eot), dim=2)

    model_depth = model_dict['model_depth']
    pred_depth, (_, _) = model_depth(in_f=in_f_depth, lengths=input_dict['lengths'])
    prediction_dict['model_depth'] = pred_depth

  return prediction_dict, in_f, missing_dict

def latent_transform(in_f):
  # Angle same = xcos, y, zsin
  if args.latent_transf == 'angle_same' or args.latent_transf == 'angle_sameP':
    if args.in_refine == 'xyz' or args.in_refine == 'dtxyz':
      angle = pt.cat((in_f[..., [3]], pt.ones(size=(in_f.shape[0], in_f.shape[1], 1)).to(device), in_f[..., [4]]), dim=2)
      in_f_ = pt.mul(in_f[..., [0, 1, 2]], angle)       # xcos, y, zsin or dt_xcos, dt_y, dt_zsin
    elif args.in_refine == 'xyz_dtxyz':
      angle = pt.cat((in_f[..., [6]], pt.ones(size=(in_f.shape[0], in_f.shape[1], 1)).to(device), in_f[..., [7]]), dim=2)
      in_f_xyz = pt.mul(in_f[..., [0, 1, 2]], angle)        # xcos, y, zsin
      in_f_dtxyz = pt.mul(in_f[..., [3, 4, 5]], angle)      # dt_xcos, dt_y, dt_zsin
      in_f_ = pt.cat((in_f_xyz, in_f_dtxyz), dim=2)

    if args.latent_transf == 'angle_sameP':
      in_f_ = pt.cat((in_f_, angle[..., [0, 2]]), dim=2)

  # Angle expand = xcos, xsin, ycos, ysin, zcos, zsin, ...
  elif args.latent_transf == 'angle_expand' or args.latent_transf == 'angle_expandP':
    if args.in_refine == 'xyz' or args.in_refine == 'dtxyz':
      angle = pt.cat((in_f[..., [3]], in_f[..., [4]]), dim=2)
      in_f_x = in_f[..., [0]] * angle       # xsin, xcos or dt_xsin, dt_xcos
      in_f_y = in_f[..., [1]] * angle       # ysin, ycos or dt_ysin, dt_ycos
      in_f_z = in_f[..., [2]] * angle       # zsin, zcos or dt_zsin, dt_zcos
      in_f_ = pt.cat((in_f_x, in_f_y, in_f_z), dim=2)
    elif args.in_refine == 'xyz_dtxyz':
      angle = pt.cat((in_f[..., [6]], in_f[..., [7]]), dim=2)
      in_f_x = in_f[..., [0]] * angle       # xsin, xcos
      in_f_y = in_f[..., [1]] * angle       # ysin, ycos
      in_f_z = in_f[..., [2]] * angle       # zsin, zcos
      in_f_dtx = in_f[..., [3]] * angle     # dt_xsin, dt_xcos
      in_f_dty = in_f[..., [4]] * angle     # dt_ysin, dt_ycos
      in_f_dtz = in_f[..., [5]] * angle     # dt_zsin, dt_zcos
      in_f_ = pt.cat((in_f_x, in_f_y, in_f_z, in_f_dtx, in_f_dty, in_f_dtz), dim=2)

    if args.latent_transf == 'angle_expandP':
      in_f_ = pt.cat((in_f_, angle), dim=2)

  return in_f_

def refinement(model_dict, gt_dict, cam_params_dict, pred_xyz, optimize, pred_dict):
  ######################################
  ############# REFINEMENT #############
  ######################################
  # print(pred_xyz.shape)
  # print(gt_dict['xyz'].shape)
  # print(gt_dict['xyz'])
  # exit()

  if args.ipl is not None:
    pred_xyz = gt_dict['xyz'][..., [0, 1, 2]]

  features_indexing = 3
  if 'eot' in args.pipeline:
    features_indexing = 4

  for idx in range(args.n_refinement):
    ####################################
    ########### N-REFINEMENT ###########
    ####################################
    '''
    Theres' 3 types of input and 2 types of output
    INPUT : 1. XYZ
            2. dXYZ (in time domain)
            3. dXYZ (in position domain)
    OUTPUT :  1. dXYZ (in time domain)
              2. dXYZ (in position domain)
    '''
    ####################################
    ############ INPUT PREP ############
    ####################################
    if args.in_refine == 'xyz':
      # xyz
      if args.optimize == 'refinement' or args.optimize == 'both':
        in_f = pt.cat((pred_xyz, gt_dict['xyz'][..., features_indexing:]), dim=2)
      else :
        in_f = pred_xyz
      # lengths
      lengths = gt_dict['lengths']
    elif args.in_refine == 'dtxyz':
      # dtxyz
      pred_xyz_delta = pred_xyz[:, :-1, :] - pred_xyz[:, 1:, :]
      if args.optimize == 'refinement' or args.optimize == 'both':
        in_f = pt.cat((pred_xyz_delta, gt_dict['xyz'][:, 1:, features_indexing:]), dim=2)
      else:
        in_f = pred_xyz_delta
      # lengths
      lengths = gt_dict['lengths']-1
    elif args.in_refine =='xyz_dtxyz':
      # dtxyz
      pred_xyz_delta = pred_xyz[:, :-1, :] - pred_xyz[:, 1:, :]
      pred_xyz_delta = pt.cat((pred_xyz_delta, pred_xyz_delta[:, [-1], :]), dim=1)
      pred_xyz_delta = utils_func.duplicate_at_length(seq=pred_xyz_delta, lengths=gt_dict['lengths'])
      # xyz & dtxyz & latent
      if args.optimize == 'refinement' or args.optimize == 'both':
        in_f = pt.cat((pred_xyz, pred_xyz_delta, gt_dict['xyz'][:, :, features_indexing:]), dim=2)
      else :
        in_f = pt.cat((pred_xyz, pred_xyz_delta), dim=2)

      # lengths
      lengths = gt_dict['lengths']

    if args.latent_transf is not None:
      in_f = latent_transform(in_f=in_f)
    # Prediction
    if 'encoder' in args.pipeline:
      model_encoder = model_dict['model_encoder']
      in_f = model_encoder(in_f)
    model_refinement = model_dict['model_refinement_{}'.format(idx)]
    pred_refinement, (_, _) = model_refinement(in_f=in_f, lengths=lengths)

    ####################################
    ########### OUTPUT PREP ############
    ####################################
    if args.out_refine == 'dtxyz_cumsum':
      # Cummulative from t=0
      if args.in_refine == 'xyz' or args.in_refine == 'xyz_dtxyz':
        pred_xyz = pt.cumsum(pt.cat((pred_xyz[:, [0], :], pred_refinement[:, :-1, [0, 1, 2]]), dim=1), dim=1)
      elif args.in_refine == 'dtxyz':
        pred_xyz = pt.cumsum(pt.cat((pred_xyz[:, [0], :], pred_refinement), dim=1), dim=1)

    elif args.out_refine == 'dtxyz_consec':
      # Consecutive from (t-1) + dt
      if args.in_refine == 'xyz' or args.in_refine == 'xyz_dtxyz':
        pred_xyz = pt.cat((pred_xyz[:, [0], :], (pred_xyz[:, :-1, :] + pred_refinement[:, :-1, [0, 1, 2]])), dim=1)
      elif args.in_refine == 'dtxyz':
        pred_xyz = pt.cat((pred_xyz[:, [0], :], (pred_xyz[:, :-1, :] + pred_refinement[..., [0, 1, 2]])), dim=1)

    elif args.out_refine == 'xyz':
      if args.in_refine == 'xyz' or args.in_refine == 'xyz_dtxyz':
        pred_xyz = pred_xyz + pred_refinement
      elif args.in_refine == 'dtxyz':
        pred_xyz = pt.cat((pred_xyz[:, [0], :], pred_xyz[:, :-1, :] + pred_refinement[..., [0, 1, 2]]), dim=1)

    elif args.out_refine =='xyz_residual':
      if args.in_refine == 'dtxyz':
        pred_xyz = pt.cat((pred_xyz[:, [0], :], (pred_xyz[:, :-1, :] + pred_refinement[..., [0, 1, 2]])), dim=1)
      elif args.in_refine == 'xyz' or args.in_refine == 'xyz_dtxyz':
        pred_xyz = pred_refinement


  return pred_xyz

def optimization_refinement(model_dict, gt_dict, cam_params_dict, pred_xyz, optimize, pred_dict):
  ###################################################
  ############# OPTIMIZATION REFINEMENT #############
  ###################################################
  '''
  Determined the latent dimension
  1. dim = 0 when
    -   No latent was fed
  2. dim > 0 when
    -   Latent was fed
    -   Encoder was used
  '''
  if args.ipl is not None:
    pred_xyz = gt_dict['xyz'][..., [0, 1, 2]] #+ pt.randn(pred_xyz.shape).to(device)/10

  # Initial the latent size
  features_indexing = 3
  if 'eot' in args.pipeline:
    features_indexing = 4
  if 'encoder' in args.pipeline:
    model_key = 'model_encoder'
  else:
    model_key = 'model_refinement_0'

  if args.latent_transf is None:
    latent_size = model_dict[model_key].input_size - features_indexing + 1 # Initial the latent size
  else:
    _, temp_size = utils_func.latent_transform_size(0, 0)
    latent_size = model_dict[model_key].input_size - features_indexing + 1 - temp_size

  # Optimizer
  trajectory_optimizer = TrajectoryOptimizationRefinement(model_dict=model_dict, gt_dict=gt_dict, cam_params_dict=cam_params_dict, latent_size=latent_size, n_refinement=args.n_refinement, pred_dict=pred_dict, latent_code=args.latent_code, latent_transf=args.latent_transf)
  # Initialize
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
    # xyz & dtxyz
    in_f = pt.cat((pred_xyz, pred_xyz_delta), dim=2)
    # lengths
    lengths = gt_dict['lengths']

  # Create latent
  trajectory_optimizer.construct_latent(lengths)
  latent_optimized = trajectory_optimizer.update_latent(lengths)
  for name, param in trajectory_optimizer.named_parameters():
    print('{}, {}, {}'.format(name, param, param.grad))
  # Optimizer
  optimizer = pt.optim.Adam(trajectory_optimizer.parameters(), lr=1)
  # optimizer = pt.optim.SGD(trajectory_optimizer.parameters(), lr=10)
  # optimizer = pt.optim.LBFGS(trajectory_optimizer.parameters(), lr=100)
  lr_scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
  trajectory_optimizer.train()
  t = tqdm(range(150), desc='Optimizing...', leave=True)
  for i in t:
    # def closure():
    optimizer.zero_grad()

    ####################################
    ########### N-REFINEMENT ###########
    ####################################
    '''
    Theres' 3 types of input and 2 types of output
    INPUT : 1. XYZ
            2. dXYZ (in time domain)
            3. dXYZ (in position domain)
    OUTPUT :  1. dXYZ (in time domain)
              2. dXYZ (in position domain)
    '''
    # Prediction
    model_encoder = None
    if 'encoder' in args.pipeline:
      model_encoder = model_dict['model_encoder']
    pred_refinement = trajectory_optimizer(in_f=in_f.detach().clone(), lengths=lengths, model_encoder=model_encoder)

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
        pred_xyz_optimized = pred_xyz.detach().clone() + pred_refinement
      elif args.in_refine == 'dtxyz':
        pred_xyz_optimized = pt.cat((pred_xyz[:, [0], :].detach().clone(), pred_xyz[:, 1:, :].detach().clone() + pred_refinement), dim=1)

    elif args.out_refine == 'xyz_residual':
      if args.in_refine == 'dtxyz':
        pred_xyz_optimized = pt.cat((pred_xyz[:, [0], :], (pred_xyz[:, :-1, :] + pred_refinement[..., [0, 1, 2]])), dim=1)
      if args.in_refine == 'xyz' or args.in_refine == 'xyz_dtxyz':
        pred_xyz_optimized = pred_refinement

    ###########################################
    ########### OPTIMIZATION LOSS #############
    ###########################################
    optimization_loss = calculate_optimization_loss(optimized_xyz=pred_xyz_optimized, gt_dict=gt_dict, cam_params_dict=cam_params_dict)
    for param_group in optimizer.param_groups:
      lr = param_group['lr']
    if lr < 1e-5: break

    t.set_description("Optimizing... (Loss = {}, LR = {})".format(optimization_loss, lr))
    t.refresh()
    lr_scheduler.step(optimization_loss)
    optimization_loss.backward()
    # for name, param in trajectory_optimizer.named_parameters():
      # print("Name : ", name)
      # print("PARAM : ", param)
      # print("GRAD : ", param.grad)

    optimizer.step()


  if pt.max(lengths) < pt.max(gt_dict['lengths']):
    latent_optimized = trajectory_optimizer.update_latent(lengths=lengths+1)
    latent_optimized = trajectory_optimizer.manipulate_latent(latent=latent_optimized)
  else:
    latent_optimized = trajectory_optimizer.update_latent(lengths=lengths)
    latent_optimized = trajectory_optimizer.manipulate_latent(latent=latent_optimized)

  return pred_xyz_optimized, latent_optimized

def optimization_depth(model_dict, gt_dict, cam_params_dict, optimize, input_dict, pred_dict):
  # Add noise on the fly
  in_f = input_dict['input'][..., [0, 1]].clone()
  if args.noise:
    in_f, missing_dict = utils_func.add_noise(input_trajectory=in_f[..., [0, 1]].clone(), startpos=input_dict['startpos'][..., [0, 1]], lengths=input_dict['lengths'])
  else:
    missing_dict = None

  # pred_dict = {}
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

  in_f = pt.cat((in_f, pred_eot), dim=2)

  features_indexing = 3
  if 'eot' in args.pipeline:
    features_indexing = 4

  latent_size = model_dict['model_depth'].input_size - features_indexing + 1 # Initial the latent size

  # Initialization
  trajectory_optimizer = TrajectoryOptimizationDepth(model_dict=model_dict, input_dict=input_dict, cam_params_dict=cam_params_dict, latent_size=latent_size, pred_dict=pred_dict, latent_code=args.latent_code)
  lengths = input_dict['lengths']
  trajectory_optimizer.construct_latent(lengths)
  latent_optimized = trajectory_optimizer.update_latent(lengths)
  trajectory_optimizer.train()
  optimizer = pt.optim.Adam(trajectory_optimizer.parameters(), lr=args.lr)
  lr_scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
  t = tqdm(range(500), desc='Optimizing...', leave=True)
  for i in t:
    optimizer.zero_grad()
    pred_depth_optimized = trajectory_optimizer(in_f=in_f.detach().clone(), lengths=lengths)
    pred_dict['model_depth'] = pred_depth_optimized

    if args.bi_pred_weight:
      bi_pred_weight = pred_depth_optimized[..., [2]]
    else:
      bi_pred_weight = pt.zeros(pred_depth_optimized[..., [0]].shape)

    pred_depth_cumsum, input_uv_cumsum = utils_cummulative.cummulative_fn(depth=pred_depth_optimized, uv=in_f[..., [0, 1]], depth_teacher=gt_dict['o_with_f'][..., [0]], startpos=input_dict['startpos'], lengths=input_dict['lengths'], eot=pred_eot, cam_params_dict=cam_params_dict, epoch=0, args=args, gt=gt_dict['xyz'][..., [0, 1, 2]], bi_pred_weight=bi_pred_weight)

    # Project the (u, v, depth) to world space
    pred_xyz_optimized = pt.stack([utils_transform.projectToWorldSpace(uv=input_uv_cumsum[i], depth=pred_depth_cumsum[i], cam_params_dict=cam_params_dict, device=device) for i in range(input_uv_cumsum.shape[0])])

    # Prediction
    if 'encoder' in args.pipeline:
      model_encoder = model_dict['model_encoder']
      pred_xyz_optimized = model_encoder(pred_xyz_optimized)

    if 'refinement' in args.pipeline:
      for idx in range(args.n_refinement):
        model_refinement = model_dict['model_refinement_{}'.format(idx)]
        pred_xyz_refined, (_, _) = model_refinement(in_f=pred_xyz_optimized, lengths=lengths+1)

      optimization_loss = calculate_optimization_loss(optimized_xyz=pred_xyz_refined, gt_dict=gt_dict, cam_params_dict=cam_params_dict)
    else:
      optimization_loss = calculate_optimization_loss(optimized_xyz=pred_xyz_optimized, gt_dict=gt_dict, cam_params_dict=cam_params_dict)

    for param_group in optimizer.param_groups:
      lr = param_group['lr']
    if lr < 1e-5: break

    t.set_description("Optimizing... (Loss = {}, LR = {})".format(optimization_loss, lr))
    t.refresh()

    lr_scheduler.step(optimization_loss)
    optimization_loss.backward()
    for name, param in trajectory_optimizer.named_parameters():
      print("Name : ", name)
      print("PARAM : ", param)
      print("GRAD : ", param.grad)
    optimizer.step()

  for name, param in trajectory_optimizer.named_parameters():
    print("Name : ", name)
    print("PARAM : ", param)
    print("GRAD : ", param.grad)

  latent_optimized = trajectory_optimizer.update_latent(lengths=lengths)
  latent_optimized = trajectory_optimizer.manipulate_latent(latent=latent_optimized)
  latent_optimized = pt.cat((latent_optimized[:, [0], :], latent_optimized), dim=1)

  return pred_xyz_optimized, latent_optimized, pred_dict, input_uv_cumsum, pred_depth_cumsum

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
  optimization_loss = below_ground_loss + multiview_loss + gravity_loss #+ multiview_loss # + trajectory_loss
  return optimization_loss

def train_mode(model_dict):
  for model in model_dict.keys():
    model_dict[model].train()

def eval_mode(model_dict):
  for model in model_dict.keys():
    model_dict[model].eval()

def calculate_loss(pred_xyz, input_dict, gt_dict, cam_params_dict, pred_dict, missing_dict, annealing_weight=None, pred_xyz_refined=None):
  # print(pred_dict.keys())
  # exit()
  # Calculate loss term
  ######################################
  ############# Trajectory #############
  ######################################
  if 'depth' in args.pipeline or 'refinement' in args.pipeline:
    trajectory_loss = utils_loss.TrajectoryLoss(pred=pred_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
    gravity_loss = utils_loss.GravityLoss(pred=pred_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])
    below_ground_loss = utils_loss.BelowGroundPenalize(pred=pred_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])

    if args.annealing:
      trajectory_loss_refined = utils_loss.TrajectoryLoss(pred=pred_xyz_refined[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1, 2]], lengths=gt_dict['lengths'])

      trajectory_loss = annealing_weight * trajectory_loss + trajectory_loss_refined * (1 - annealing_weight)
  else:
    trajectory_loss = pt.tensor(0.).to(device)
    gravity_loss = pt.tensor(0.).to(device)
    below_ground_loss = pt.tensor(0.).to(device)

  ######################################
  ################ Flag ################
  ######################################
  if 'eot' in args.pipeline and args.env == 'unity':
    pred_eot = pred_dict['model_flag']
    eot_loss = utils_loss.EndOfTrajectoryLoss(pred=pred_eot, gt=gt_dict['o_with_f'][..., [1]], mask=input_dict['mask'][..., [2]], lengths=input_dict['lengths'], startpos=input_dict['startpos'][..., [2]], flag='Train')
  else:
    eot_loss = pt.tensor(0.).to(device)

  ######################################
  ############ Reprojection ############
  ######################################

  if len(args.multiview_loss) > 0:
    multiview_loss = utils_loss.MultiviewReprojectionLoss(pred=pred_xyz[..., [0, 1, 2]], gt=gt_dict['xyz'][..., [0, 1, 2]], mask=gt_dict['mask'][..., [0, 1]], lengths=gt_dict['lengths'], cam_params_dict=cam_params_dict)
  else:
    multiview_loss = pt.tensor(0.).to(device)


  ######################################
  ########### Interpolation ############
  ######################################
  if 'uv' in args.pipeline:
    uv_pred = pt.cat((pt.unsqueeze(input_dict['input'][:, [0], [0, 1]], dim=1), pred_dict['model_uv'][:, :-1, [0, 1]]), dim=1)
    interpolation_loss = utils_loss.InterpolationLoss(uv_gt=input_dict['input'][..., [0, 1]], uv_pred=uv_pred, mask=input_dict['mask'][..., [0, 1]], lengths=input_dict['lengths'])
  else:
    interpolation_loss = pt.tensor(0.).to(device)

  ######################################
  ############### Depth ################
  ######################################
  if 'depth' in args.pipeline:
    if args.bi_pred_ramp:
      depth_loss_fw = utils_loss.DepthLoss(pred=pred_dict['model_depth'][..., [0]], gt=gt_dict['o_with_f'][..., [0]], lengths=input_dict['lengths'], mask=input_dict['mask'][..., [0]])
      depth_loss_bw = utils_loss.DepthLoss(pred=pred_dict['model_depth'][..., [1]], gt=-gt_dict['o_with_f'][..., [0]], lengths=input_dict['lengths'], mask=input_dict['mask'][..., [0]])
      depth_loss = depth_loss_fw + depth_loss_bw
  else:
    depth_loss = pt.tensor(0.).to(device)

  # Sum up all loss 
  loss = trajectory_loss + eot_loss + gravity_loss + below_ground_loss + multiview_loss + interpolation_loss + depth_loss * 100
  loss_dict = {"Trajectory Loss":trajectory_loss.item(), "EndOfTrajectory Loss":eot_loss.item(), "Gravity Loss":gravity_loss.item(), "BelowGroundPenalize Loss":below_ground_loss.item(), "MultiviewReprojection Loss":multiview_loss.item(), "Interpolation Loss":interpolation_loss.item(), "Depth Loss":depth_loss.item() * 100}

  return loss_dict, loss
