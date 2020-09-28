import torch as pt
import numpy as np

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
else:
  device = pt.device('cpu')

def cummulative_fn(xyz, xyz_teacher, uv, startpos, lengths, eot, cam_params_dict, epoch, args):
  '''Cummulative modules with 4 diiferent ways
  1. Decumulate
  2. Teacher forcing all samples
  3. Teacher forcing some samples
  4. Normal cummulative
  '''
  # De-accumulate module
  # (This step we get the displacement of xyz by input the displacement of u and v)
  # Apply cummulative summation to output using cumsum_trajectory function
  if args.decumulate and epoch > args.start_decumulate:
    xyz_cumsum, uv_cumsum, fail = cumsum_decumulate_trajectory(xyz=xyz, uv=uv[..., [0, 1]], trajectory_startpos=startpos, lengths=lengths, eot=eot, cam_params_dict=cam_params_dict)
    if fail:
      xyz_cumsum, uv_cumsum = cumsum_trajectory(xyz=xyz, uv=uv[..., [0, 1]], trajectory_startpos=startpos)

  elif args.teacherforcing_xyz:
    xyz_cumsum, uv_cumsum = cumsum_teacherforcing_trajectory(xyz=xyz, xyz_teacher=xyz_teacher, uv=uv[..., [0, 1]], trajectory_startpos=startpos)

  elif args.teacherforcing_mixed:
    factor = np.random.uniform(low=0.3, high=0.6)
    n_teacherforcing = int(args.batch_size * factor)
    teacher_idx = np.random.choice(a=args.batch_size, size=(n_teacherforcing,), replace=False)
    xyz_cumsum, uv_cumsum = cumsum_trajectory(xyz=xyz, uv=uv[..., [0, 1]], trajectory_startpos=startpos)
    xyz_cumsum[teacher_idx, ...], _ = cumsum_teacherforcing_trajectory(xyz=xyz[teacher_idx, ...], xyz_teacher=xyz_teacher[teacher_idx, ...], uv=uv[teacher_idx][..., [0, 1]], trajectory_startpos=startpos[teacher_idx])
  else:
    xyz_cumsum, uv_cumsum = cumsum_trajectory(xyz=xyz, uv=uv[..., [0, 1]], trajectory_startpos=startpos)

  return xyz_cumsum, uv_cumsum

def get_plane_normal():
  a = pt.tensor([32., 0., 19.])
  b = pt.tensor([32., 0., -31.])
  c = pt.tensor([-28., 0., 19.])
  plane_normal = pt.cross(b-a, c-a)
  return plane_normal.to(device)

def raycasting(reset_idx, uv, lengths, xyz, cam_params_dict, plane_normal):
  screen_width = cam_params_dict['width']
  screen_height = cam_params_dict['height']
  I_inv = cam_params_dict['I_inv']
  E_inv = cam_params_dict['E_inv']
  # print(reset_idx, uv, lengths, xyz)
  camera_center = E_inv[:-1, -1]
  # Ray casting
  transformation = pt.inverse(pt.inverse(I_inv) @ pt.inverse(E_inv))   # Inverse(Intrinsic @ Extrinsic)
  uv = pt.cat((uv[reset_idx[0], :], pt.ones(uv[reset_idx[0], :].shape).to(device)), dim=-1) # reset_idx[0] is 
  uv[:, 0] = ((uv[:, 0]/screen_width) * 2) - 1
  uv[:, 1] = ((uv[:, 1]/screen_height) * 2) - 1
  ndc = (uv @ transformation.t()).to(device)
  ray_direction = ndc[:, :-1] - camera_center
  # xyz that intersect the pitch
  plane_point = pt.tensor([32, 0, 19]).to(device)
  distance = camera_center - plane_point
  normalize = pt.tensor([(pt.dot(distance, plane_normal)/pt.dot(ray_direction[i], plane_normal)) for i in range(ray_direction.shape[0])]).view(-1, 1).to(device)
  intersect_pos = pt.cat(((camera_center - (ray_direction * normalize)), pt.ones(ray_direction.shape[0], 1).to(device)), dim=-1)
  return intersect_pos[..., [0, 1, 2]]

def split_cumsum(reset_idx, length, startpos, reset_xyz, xyz, eot):
  '''
  1. This will split the xyz displacement from reset_idx into a chunk. (Ignore the prediction where the EOT=1 in prediction variable. Because we will cast the ray to get that reset xyz instead of cumsum to get it.)
  2. Perform cumsum seperately of each chunk.
  3. Concatenate all u, v, xyz together and replace with the current one. (Need to replace with padding for masking later on.)
  '''
  reset_idx -= 1
  reset_xyz = pt.cat((pt.unsqueeze(startpos[0][[2, 3, 4]], dim=0), reset_xyz))
  max_len = pt.tensor(xyz.shape[0]).view(-1, 1).to(device)
  reset_idx = pt.cat((pt.zeros(1).type(pt.cuda.LongTensor).view(-1, 1).to(device), reset_idx.view(-1, 1)))
  if reset_idx[-1] != xyz.shape[0] and reset_idx.shape[0] > 1:
    reset_idx = pt.cat((reset_idx, max_len))
  xyz_chunk = [xyz[start:end] if start == 0 else xyz[start+1:end] for start, end in zip(reset_idx, reset_idx[1:])]
  xyz_chunk = [pt.cat((pt.unsqueeze(reset_xyz[i], dim=0), xyz_chunk[i])) for i in range(len(xyz_chunk))]
  xyz_chunk_cumsum = [pt.cumsum(each_xyz_chunk, dim=0) for each_xyz_chunk in xyz_chunk]
  xyz_chunk = pt.cat(xyz_chunk_cumsum)
  return xyz_chunk

def cumsum_decumulate_trajectory(xyz, uv, trajectory_startpos, lengths, eot, cam_params_dict):
  # print(xyz.shape, uv.shape, trajectory_startpos.shape, eot.shape)
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
  # Reset the xyz when eot == 1
  plane_normal = get_plane_normal()

  eot_all = pt.stack([pt.cat([trajectory_startpos[i][:, [5]].view(-1, 1), eot[i]]) for i in range(trajectory_startpos.shape[0])])
  reset_idx = [pt.where((eot_all[i][:lengths[i]+1]) == 1.) for i in range(eot_all.shape[0])]
  check_reset_idx = pt.sum(pt.tensor([(reset_idx[i][0].nelement() == 0) for i in range(eot_all.shape[0])])) == len(reset_idx)
  if check_reset_idx == True:
    return None, None, True
  reset_xyz = [raycasting(reset_idx=reset_idx[i], xyz=xyz[i], uv=uv_cumsum[i], lengths=lengths[i], cam_params_dict=cam_params_dict, plane_normal=plane_normal) for i in range(trajectory_startpos.shape[0])]
  # output : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  xyz_cumsum = [split_cumsum(reset_idx=reset_idx[i][0]+1, length=lengths[i], reset_xyz=reset_xyz[i], startpos=trajectory_startpos[i], xyz=xyz[i], eot=eot_all[i]) for i in range(trajectory_startpos.shape[0])]
  xyz_cumsum = pt.stack(xyz_cumsum, dim=0)
  return xyz_cumsum, uv_cumsum, False

def cumsum_teacherforcing_trajectory(xyz, xyz_teacher, uv, trajectory_startpos):
  '''
  Perform a cummulative summation to the output
  Argument :
  1. xyz : The displacement from the network with shape = (batch_size, sequence_length,)
  2. uv : The input_trajectory (displacement of u, v) with shape = (batch_size, sequence_length, 2)
  3. trajectory_startpos : The start position of input trajectory with shape = (batch_size, 1, )
  Output :
  1. output : concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1,)
  2. uv_cumsum : u, v by concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1, 2)
  '''
  # Teacher forcing use the ground truth xyz
  xyz_teacher = pt.stack([pt.cat([trajectory_startpos[i][:, [2, 3, 4]], xyz_teacher[i][:, [0, 1, 2]]]) for i in range(trajectory_startpos.shape[0])])
  xyz_teacher = pt.cumsum(xyz_teacher, dim=1)
  # Apply cummulative summation to output
  # trajectory_cumsum : concat with startpos and stack back to (batch_size, sequence_length+1, 2)
  uv_cumsum = pt.stack([pt.cat([trajectory_startpos[i][:, [0, 1]], uv[i].clone().detach()]) for i in range(trajectory_startpos.shape[0])])
  # trajectory_cumsum : perform cumsum along the sequence_length axis
  uv_cumsum = pt.cumsum(uv_cumsum, dim=1)
  # output : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  xyz_cumsum = xyz + xyz_teacher[:, :-1, :]
  xyz_cumsum = pt.stack([pt.cat([trajectory_startpos[i][:, [2, 3 , 4]], xyz_cumsum[i][:, [0, 1, 2]]]) for i in range(trajectory_startpos.shape[0])])
  # output : perform cumsum along the sequence_length axis
  return xyz_cumsum, uv_cumsum

def cumsum_trajectory(xyz, uv, trajectory_startpos):
  '''
  Perform a cummulative summation to the output
  Argument :
  1. xyz : The displacement from the network with shape = (batch_size, sequence_length,)
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
  output = pt.stack([pt.cat([trajectory_startpos[i][:, [2, 3, 4]], xyz[i]]) for i in range(trajectory_startpos.shape[0])])
  # output : perform cumsum along the sequence_length axis
  xyz = pt.cumsum(output, dim=1)
  return xyz, uv_cumsum
