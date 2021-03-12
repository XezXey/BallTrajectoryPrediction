import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.realpath('../..'))
import utils.transformation as transformation
import utils.utils_func as utils_func
args = None

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
else:
  device = pt.device('cpu')

def share_args(a):
  global args
  args = a

def cummulative_fn(depth, depth_teacher, uv, startpos, lengths, eot, cam_params_dict, epoch, args, gt=None, bi_pred_weight=None):
  '''Cummulative modules with 4 diiferent ways
  1. Decumulate
  2. Teacher forcing all samples
  3. Teacher forcing some samples
  4. Normal cummulative
  '''
  # De-accumulate module
  # (This step we get the displacement of depth by input the displacement of u and v)
  # Apply cummulative summation to output using cumsum_trajectory function
  cam_params_dict = cam_params_dict['main']
  if args.decumulate and epoch > args.start_decumulate:
    depth_cumsum, uv_cumsum, fail = cumsum_decumulate_trajectory(depth=depth, uv=uv[..., [0, 1]], trajectory_startpos=startpos, lengths=lengths, eot=eot, cam_params_dict=cam_params_dict, args=args)
    if fail:
      depth_cumsum, uv_cumsum = cumsum_trajectory(depth=depth, uv=uv[..., [0, 1]], trajectory_startpos=startpos[..., [0, 1, 2]])

  elif args.teacherforcing_depth:
    depth_cumsum, uv_cumsum = cumsum_teacherforcing_trajectory(depth=depth, depth_teacher=depth_teacher, uv=uv[..., [0, 1]], trajectory_startpos=startpos[..., [0, 1, 2]])

  elif args.teacherforcing_mixed:
    factor = np.random.uniform(low=0.3, high=0.6)
    n_teacherforcing = int(args.batch_size * factor)
    teacher_idx = np.random.choice(a=args.batch_size, size=(n_teacherforcing,), replace=False)
    depth_cumsum, uv_cumsum = cumsum_trajectory(depth=depth, uv=uv[..., [0, 1]], trajectory_startpos=startpos[..., [0, 1, 2]])
    depth_cumsum[teacher_idx, ...], _ = cumsum_teacherforcing_trajectory(depth=depth[teacher_idx, ...], depth_teacher=depth_teacher[teacher_idx, ...], uv=uv[teacher_idx][..., [0, 1]], trajectory_startpos=startpos[teacher_idx][..., [0, 1, 2]])

  elif args.bi_pred_avg:
    # Function for bidirectional prediction
    # Function that take xyz to project to get u, v, d
    _, _, d = transformation.projectToScreenSpace(world=gt, cam_params_dict=cam_params_dict, normalize=False)   # Without normalize to NDC space
    # lengths variable here is the input sequence_length and we need the positition before lengths+1 so we use lengths here
    last_d = pt.tensor([d[i][lengths[i], :] for i in range(d.shape[0])]).view(-1, 1, 1).to(device)
    # forward depth
    depth_fw_cumsum, uv_cumsum = cumsum_trajectory(depth=depth[..., [0]], uv=uv[..., [0, 1]], trajectory_startpos=startpos[..., [0, 1, 2]])
    # backward depth
    depth_bw = utils_func.reverse_masked_seq(seq=depth[..., [1]], lengths=lengths)
    depth_bw_cumsum, _ = cumsum_trajectory(depth=depth_bw[..., [0]], uv=uv[..., [0, 1]], trajectory_startpos=pt.cat((startpos[..., [0, 1]], last_d), dim=2))
    depth_bw_cumsum = utils_func.reverse_masked_seq(seq=depth_bw_cumsum[..., [0]], lengths=lengths+1)
    depth_cumsum = pt.unsqueeze(pt.mean(pt.cat((depth_fw_cumsum, depth_bw_cumsum), dim=2), dim=2), dim=2)

  elif args.bi_pred_weight:
    # Function for bidirectional prediction
    bi_pred_weight = utils_func.construct_bipred_weight(weight=bi_pred_weight, lengths=lengths)
    # Function that take xyz to project to get u, v, d
    _, _, d = transformation.projectToScreenSpace(world=gt, cam_params_dict=cam_params_dict, normalize=False)   # Without normalize to NDC space
    # lengths variable here is the input sequence_length and we need the positition before lengths+1 so we use lengths here
    last_d = pt.tensor([d[i][lengths[i], :] for i in range(d.shape[0])]).view(-1, 1, 1).to(device)
    # forward depth
    depth_fw_cumsum, uv_cumsum = cumsum_trajectory(depth=depth[..., [0]], uv=uv[..., [0, 1]], trajectory_startpos=startpos[..., [0, 1, 2]])
    # backward depth
    depth_bw = utils_func.reverse_masked_seq(seq=depth[..., [1]], lengths=lengths)
    depth_bw_cumsum, _ = cumsum_trajectory(depth=depth_bw[..., [0]], uv=uv[..., [0, 1]], trajectory_startpos=pt.cat((startpos[..., [0, 1]], last_d), dim=2))
    depth_bw_cumsum = utils_func.reverse_masked_seq(seq=depth_bw_cumsum[..., [0]], lengths=lengths+1)
    # print(bi_pred_weight.shape)
    # print(pt.cat((depth_fw_cumsum, depth_bw_cumsum), dim=2).shape)
    # exit()
    depth_cumsum = pt.sum(pt.cat((depth_fw_cumsum, depth_bw_cumsum), dim=2) * bi_pred_weight, dim=2)

  elif args.bw_pred:
    # Function for backward prediction
    # Function that take xyz to project to get u, v, d
    _, _, d = transformation.projectToScreenSpace(world=gt, cam_params_dict=cam_params_dict, normalize=False)   # Without normalize to NDC space
    # lengths variable here is the input sequence_length and we need the positition before lengths+1 so we use lengths here
    last_d = pt.tensor([d[i][lengths[i], :] for i in range(d.shape[0])]).view(-1, 1, 1).to(device)
    # backward depth
    depth_bw = utils_func.reverse_masked_seq(seq=depth[..., [0]], lengths=lengths)
    depth_bw_cumsum, uv_cumsum = cumsum_trajectory(depth=depth_bw[..., [0]], uv=uv[..., [0, 1]], trajectory_startpos=pt.cat((startpos[..., [0, 1]], last_d), dim=2))
    depth_bw_cumsum = utils_func.reverse_masked_seq(seq=depth_bw_cumsum[..., [0]], lengths=lengths+1)
    depth_cumsum = depth_bw_cumsum

  elif args.bi_pred_ramp:
    # Function for bidirectional prediction
    bi_pred_ramp = utils_func.construct_bipred_ramp(weight_template=bi_pred_weight, lengths=lengths)
    # Function that take xyz to project to get u, v, d
    _, _, d = transformation.projectToScreenSpace(world=gt, cam_params_dict=cam_params_dict, normalize=False)   # Without normalize to NDC space
    # lengths variable here is the input sequence_length and we need the positition before lengths+1 so we use lengths here
    last_d = pt.tensor([d[i][lengths[i], :] for i in range(d.shape[0])]).view(-1, 1, 1).to(device)
    # forward depth
    depth_fw_cumsum, uv_cumsum = cumsum_trajectory(depth=depth[..., [0]], uv=uv[..., [0, 1]], trajectory_startpos=startpos[..., [0, 1, 2]])
    # backward depth
    depth_bw = utils_func.reverse_masked_seq(seq=depth[..., [1]], lengths=lengths)
    depth_bw_cumsum, _ = cumsum_trajectory(depth=depth_bw[..., [0]], uv=uv[..., [0, 1]], trajectory_startpos=pt.cat((startpos[..., [0, 1]], last_d), dim=2))
    depth_bw_cumsum = utils_func.reverse_masked_seq(seq=depth_bw_cumsum[..., [0]], lengths=lengths+1)
    depth_cumsum = pt.unsqueeze(pt.sum(pt.cat((depth_fw_cumsum, depth_bw_cumsum), dim=2) * bi_pred_ramp, dim=2), dim=2)

  elif args.si_pred_ramp:
    # Function for bidirectional prediction
    bi_pred_ramp = utils_func.construct_bipred_ramp(weight_template=bi_pred_weight, lengths=lengths)
    # Function that take xyz to project to get u, v, d
    _, _, d = transformation.projectToScreenSpace(world=gt, cam_params_dict=cam_params_dict, normalize=False)   # Without normalize to NDC space
    # lengths variable here is the input sequence_length and we need the positition before lengths+1 so we use lengths here
    last_d = pt.tensor([d[i][lengths[i], :] for i in range(d.shape[0])]).view(-1, 1, 1).to(device)
    # forward depth
    depth_fw_cumsum, uv_cumsum = cumsum_trajectory(depth=depth[..., [0]], uv=uv[..., [0, 1]], trajectory_startpos=startpos[..., [0, 1, 2]])
    # backward depth
    depth_bw = utils_func.reverse_masked_seq(seq=-depth[..., [0]].clone(), lengths=lengths)
    depth_bw_cumsum, _ = cumsum_trajectory(depth=depth_bw[..., [0]], uv=uv[..., [0, 1]], trajectory_startpos=pt.cat((startpos[..., [0, 1]], last_d), dim=2))
    depth_bw_cumsum = utils_func.reverse_masked_seq(seq=depth_bw_cumsum[..., [0]], lengths=lengths+1)
    depth_cumsum = pt.unsqueeze(pt.sum(pt.cat((depth_fw_cumsum, depth_bw_cumsum), dim=2) * bi_pred_ramp, dim=2), dim=2)

  else:
    depth_cumsum, uv_cumsum = cumsum_trajectory(depth=depth, uv=uv[..., [0, 1]], trajectory_startpos=startpos[..., [0, 1, 2]])

  return depth_cumsum, uv_cumsum

def get_plane_normal():
  a = pt.tensor([32., 0., 19.])
  b = pt.tensor([32., 0., -31.])
  c = pt.tensor([-28., 0., 19.])
  plane_normal = pt.cross(b-a, c-a)
  return plane_normal.to(device)

def raycasting(reset_idx, uv, lengths, depth, cam_params_dict, plane_normal):
  screen_width = cam_params_dict['width']
  screen_height = cam_params_dict['height']
  I = cam_params_dict['I']
  I_inv = cam_params_dict['I_inv']
  E = cam_params_dict['E']
  E_inv = cam_params_dict['E_inv']
  # print(reset_idx, uv, lengths, depth)
  camera_center = E_inv[:-1, -1]
  # Ray casting
  transformation = I @ E   # Intrinsic @ Extrinsic
  transformation_inv = pt.inverse(transformation)   # Inverse(Intrinsic @ Extrinsic)
  uv = pt.cat((uv[reset_idx[0], :], pt.ones(uv[reset_idx[0], :].shape).to(device)), dim=-1) # reset_idx[0] is 
  uv[:, 0] = ((uv[:, 0]/screen_width) * 2) - 1
  uv[:, 1] = ((uv[:, 1]/screen_height) * 2) - 1
  ndc = (uv @ transformation_inv.t()).to(device)
  ray_direction = ndc[:, :-1] - camera_center
  # Depth that intersect the pitch
  plane_point = pt.tensor([32., 0., 19.]).to(device)
  distance = camera_center - plane_point
  normalize = pt.tensor([(pt.dot(distance, plane_normal)/pt.dot(ray_direction[i], plane_normal)) for i in range(ray_direction.shape[0])]).view(-1, 1).to(device)
  intersect_pos = pt.cat(((camera_center - ray_direction * normalize), pt.ones(ray_direction.shape[0], 1).to(device)), dim=-1)
  # Make all position on intersected to zero
  intersect_pos[:, 1] = 0.
  reset_depth = intersect_pos @ E.t()
  return -reset_depth[..., 2].view(-1, 1)

def split_cumsum(reset_idx, length, start_pos, reset_depth, depth, eot):
  '''
  1. This will split the depth displacement from reset_idx into a chunk. (Ignore the prediction where the EOT=1 in prediction variable. Because we will cast the ray to get that reset depth instead of cumsum to get it.)
  2. Perform cumsum seperately of each chunk.
  3. Concatenate all u, v, depth together and replace with the current one. (Need to replace with padding for masking later on.)
  '''
  # print("B4", reset_idx)
  reset_idx -= 1
  # print("AF", reset_idx)
  reset_depth = pt.cat((start_pos[0][2].view(-1, 1), reset_depth))
  max_len = pt.tensor(depth.shape[0]).view(-1, 1).to(device)
  if reset_idx[0] != 0:
    reset_idx = pt.cat((pt.zeros(1).type(pt.cuda.LongTensor).view(-1, 1).to(device), reset_idx.view(-1, 1)))
  if (reset_idx[-1] != depth.shape[0] and reset_idx.shape[0] > 1) or (len(reset_idx) == 1):
    reset_idx = pt.cat((reset_idx.view(-1, 1), max_len.view(-1, 1)))
  depth_chunk = [depth[start:end] if start == 0 else depth[start+1:end] for start, end in zip(reset_idx, reset_idx[1:])]
  depth_chunk = [pt.cat((reset_depth[i].view(-1, 1), depth_chunk[i])) for i in range(len(depth_chunk))]
  # depth_chunk_cumsum = [each_depth_chunk[0].view(-1, 1) if (idx!=len(depth_chunk)-1) else pt.cumsum(each_depth_chunk, dim=0) for idx, each_depth_chunk in enumerate(depth_chunk)]
  depth_chunk_cumsum = [pt.cumsum(each_depth_chunk, dim=0) for each_depth_chunk in depth_chunk]
  depth_chunk = pt.cat(depth_chunk_cumsum)
  # plt.plot(depth_chunk.cpu().detach().numpy())
  # plt.show()
  # print(reset_idx)
  return depth_chunk

def cumsum_decumulate_trajectory(depth, uv, trajectory_startpos, lengths, eot, cam_params_dict, args):
  # print(depth.shape, uv.shape, trajectory_startpos.shape, eot.shape)
  '''
  Perform a cummulative summation to the output
  Argument :
  1. output : The displacement from the network with shape = (batch_size, sequence_length,)
  2. trajectory : The input_trajectory (displacement of u, v) with shape = (batch_size, sequence_length, 2)
  3. trajectory_startpos : The start position of input trajectory with shape = (batch_size, 1, )
  Output :
  1. output : concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1,)
  2. trajectory_temp : u, v by concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1, 2)
  '''
  # Apply cummulative summation to output
  # uv_cumsum : concat with startpos and stack back to (batch_size, sequence_length+1, 2)
  uv_cumsum = pt.stack([pt.cat([trajectory_startpos[i][:, [0, 1]], uv[i].clone().detach()]) for i in range(trajectory_startpos.shape[0])])
  # uv_cumsum : perform cumsum along the sequence_length axis
  uv_cumsum = pt.cumsum(uv_cumsum, dim=1)
  # Reset the depth when eot == 1
  plane_normal = get_plane_normal()

  if args.env == 'mocap':
    eot_all = pt.stack([pt.cat([pt.tensor([0.]).to(device).view(-1, 1), eot[i]]) for i in range(trajectory_startpos.shape[0])])
  elif args.env == 'unity':
    eot_all = pt.stack([pt.cat([trajectory_startpos[i][:, [3]].view(-1, 1), eot[i]]) for i in range(trajectory_startpos.shape[0])])
  eot_all = (eot_all > 0.5).type(pt.cuda.FloatTensor)
  reset_idx = [pt.where((eot_all[i][:lengths[i]+1]) == 1.) for i in range(eot_all.shape[0])]
  check_reset_idx = pt.sum(pt.tensor([(reset_idx[i][0].nelement() == 0) for i in range(eot_all.shape[0])])) == len(reset_idx)
  if check_reset_idx == True:
    return None, None, True
  reset_depth = [raycasting(reset_idx=reset_idx[i], depth=depth[i], uv=uv_cumsum[i], lengths=lengths[i], cam_params_dict=cam_params_dict, plane_normal=plane_normal) for i in range(trajectory_startpos.shape[0])]
  # output : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  depth_cumsum = [split_cumsum(reset_idx=reset_idx[i][0]+1, length=lengths[i], reset_depth=reset_depth[i], start_pos=trajectory_startpos[i], depth=depth[i], eot=eot_all[i]) for i in range(trajectory_startpos.shape[0])]
  depth_cumsum = pt.stack(depth_cumsum, dim=0)
  return depth_cumsum, uv_cumsum, False

def cumsum_teacherforcing_trajectory(depth, depth_teacher, uv, trajectory_startpos):
  '''
  Perform a cummulative summation to the output
  Argument :
  1. depth : The displacement from the network with shape = (batch_size, sequence_length,)
  2. uv : The input_trajectory (displacement of u, v) with shape = (batch_size, sequence_length, 2)
  3. trajectory_startpos : The start position of input trajectory with shape = (batch_size, 1, )
  Output :
  1. output : concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1,)
  2. uv_cumsum : u, v by concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1, 2)
  '''
  # Teacher forcing use the ground truth depth
  depth_teacher = pt.stack([pt.cat([trajectory_startpos[i][:, -1], depth_teacher[i][:, 0]]) for i in range(trajectory_startpos.shape[0])])
  depth_teacher = pt.cumsum(depth_teacher, dim=1).unsqueeze(dim=-1)
  # Apply cummulative summation to output
  # trajectory_cumsum : concat with startpos and stack back to (batch_size, sequence_length+1, 2)
  uv_cumsum = pt.stack([pt.cat([trajectory_startpos[i][:, :2], uv[i].clone().detach()]) for i in range(trajectory_startpos.shape[0])])
  # trajectory_cumsum : perform cumsum along the sequence_length axis
  uv_cumsum = pt.cumsum(uv_cumsum, dim=1)
  # output : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  depth_cumsum = depth + depth_teacher[:, :-1, :]
  depth_cumsum = pt.stack([pt.cat([trajectory_startpos[i][:, -1], depth_cumsum[i][:, 0]]) for i in range(trajectory_startpos.shape[0])])
  # output = pt.stack([pt.cat([trajectory_startpos[i][:, -1].view(-1, 1), depth[i]]) for i in range(trajectory_startpos.shape[0])])
  # output : perform cumsum along the sequence_length axis
  return pt.unsqueeze(depth_cumsum, dim=-1), uv_cumsum

def cumsum_trajectory(depth, uv, trajectory_startpos):
  '''
  Perform a cummulative summation to the output
  Argument :
  1. depth : The displacement from the network with shape = (batch_size, sequence_length,)
  2. uv : The input_trajectory (displacement of u, v) with shape = (batch_size, sequence_length, 2)
  3. trajectory_startpos : The start position of input trajectory with shape = (batch_size, 1, )
  Output :
  1. output : concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1,)
  2. uv_cumsum : u, v by concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1, 2)
  '''
  # Apply cummulative summation to output
  # trajectory_cumsum : concat with startpos and stack back to (batch_size, sequence_length+1, 2)
  uv_cumsum = pt.stack([pt.cat([trajectory_startpos[i][:, [0, 1]], uv[i].clone().detach()]) for i in range(trajectory_startpos.shape[0])])
  # trajectory_cumsum : perform cumsum along the sequence_length axis
  uv_cumsum = pt.cumsum(uv_cumsum, dim=1)
  # output : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output = pt.stack([pt.cat([trajectory_startpos[i][:, [2]], depth[i]]) for i in range(trajectory_startpos.shape[0])])
  # output : perform cumsum along the sequence_length axis
  depth = pt.cumsum(output, dim=1)
  return depth, uv_cumsum
