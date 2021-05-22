import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import argparse
import math

'''
Input File convention :
1. Not in a dt-space
2. Columns sorted to : x, y, z, u, v, d, eot, latent(if exists)
3. Shape input is (n_trajectory, )
4. All sample are sorted in the same order
'''

parser = argparse.ArgumentParser()
parser.add_argument('--cmps', nargs='+', default=[])
parser.add_argument('--sample_idx', type=int, required=False, default=None)
parser.add_argument('--selection_2d', type=str, default='xyz')
parser.add_argument('--path_name', type=str, required=True)
parser.add_argument('--folder_name', type=str, required=True)
parser.add_argument('--ours', nargs='+', default=[None])
parser.add_argument('--gt', type=str, default=None)
parser.add_argument('--latent', type=str)
parser.add_argument('--flag', type=str)
parser.add_argument('--gt_dat', type=str)
args = parser.parse_args()




angle_fn = lambda a, b: math.atan2(
    math.sqrt(
        np.dot(*([np.cross(a, b)]*2))
    ),
    np.dot(a, b)
)

def plot3d(data, col, row, fig, marker, name):
  x = data[:, 0]
  y = data[:, 1]
  z = data[:, 2]
  fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=marker, mode='markers', name=name), row=row, col=col)
  return fig

def plot2d(x, y, col, row, fig, marker, name):
  fig.add_trace(go.Scatter(x=x, y=y, marker=marker, mode='markers+lines', name=name), row=row, col=col)
  return fig

def plot_latent(latent_idx, latent_var, data, col, row, fig, marker, name):
  if args.gt_dat == 'gt':
    for i, latent in enumerate(latent_idx):
      arrow_x = np.array([data[latent, 0], data[latent, 0] + latent_var[latent, 1]])
      arrow_y = np.array([data[latent, 1], data[latent, 1]])
      arrow_z = np.array([data[latent, 2], data[latent, 2] + latent_var[latent, 0]])
      fig.add_trace(go.Scatter3d(x=arrow_x, y=arrow_y, z=arrow_z, marker=marker, mode='lines', line=dict(width=10), name=name), row=row, col=col)
  elif args.gt_dat == 'syn_gt':
    for i, latent in enumerate(latent_idx):
      arrow_x = np.array([data[latent, 0], data[latent, 0] + latent_var[i, 1]])
      arrow_y = np.array([data[latent, 1], data[latent, 1]])
      arrow_z = np.array([data[latent, 2], data[latent, 2] + latent_var[i, 0]])
      fig.add_trace(go.Scatter3d(x=arrow_x, y=arrow_y, z=arrow_z, marker=marker, mode='lines', line=dict(width=10), name=name), row=row, col=col)

  return fig


if __name__ == '__main__':
  ours_list = []
  cmps_list = []
  args.ours = ['{}/{}/{}_trajectory_prediction.npy'.format(args.path_name, args.folder_name, args.folder_name)]
  args.flag = '{}/{}/{}_trajectory_flag.npy'.format(args.path_name, args.folder_name, args.folder_name)
  args.latent = '{}/{}/{}_trajectory_latent.npy'.format(args.path_name, args.folder_name, args.folder_name)
  args.gt = '{}/{}/{}_trajectory_gt.npy'.format(args.path_name, args.folder_name, args.folder_name)

  # OURS
  if len(args.ours) > 0:
    for our in args.ours:
      ours_list.append(np.load(file=our, allow_pickle=True))

  # COMPARE
  if len(args.cmps) > 0:
    for cmp in args.cmps:
      cmps_list.append(np.load(file=cmp, allow_pickle=True))

  # GT
  gt = np.load(file=args.gt, allow_pickle=True)
  if args.latent is not None:
    latent = np.load(file=args.latent, allow_pickle=True)
  if args.flag is not None:
    flag = np.load(file=args.flag, allow_pickle=True)

  # PLOTTING 
  fig = make_subplots(rows=1, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter'}]])
  marker_dict_gt = dict(color='rgba(0, 0, 255, 1.0)', size=3)
  marker_dict_our = dict(color='rgba(255, 0, 0, 0.7)', size=3)

  marker_dict_latent_gt = dict(color='rgba(0, 0, 0, 1.0)', size=6)
  marker_dict_latent_our = dict(color='rgba(0, 255, 0, 1.0)', size=6)
  marker_dict_cmp = dict(color='rgba(0, 0, 0, 0.7)', size=3)

  marker_dict_2d = dict(color='rgba(255, 0, 0, 0.7)', size=3)

  # 3D plot
  if args.sample_idx is not None:
    idx = args.sample_idx
  else:
    idx = np.random.randint(0, gt.shape[0])

  # Plot ours
  if len(args.ours) > 0:
    for our in ours_list:
      fig = plot3d(data=our[idx], col=1, row=1, fig=fig, marker=marker_dict_our, name="Ours")

  # Plot cmps
  if len(args.cmps) > 0:
    for cmp in cmps_list:
      fig = plot3d(data=cmp[idx][0], col=1, row=1, fig=fig, marker=marker_dict_cmp, name="Cmps")

  # Plot gt
  fig = plot3d(data=gt[idx], col=1, row=1, fig=fig, marker=marker_dict_gt, name="GT")

  axes = ['X', 'Y', 'Z']
  # 2D plot
  if len(args.ours) > 0 and len(args.cmps) > 0:
    each_gt = gt[idx][0][:, [0, 1, 2]]
    seq_len = np.arange(each_gt.shape[0])
    for our in ours_list:
      each_our = our[idx][0][:, [0, 1, 2]]
      each_our_diff = np.abs(each_our - each_gt)
      for cmp in cmps_list:
        each_cmp = cmp[idx][0][:, [0, 1, 2]]
        each_cmp_diff = np.abs(each_cmp - each_gt)
        each_improv = each_our_diff - each_cmp_diff


        for axis in range(0, 3):
          if args.selection_2d == 'xyz':
            fig = plot2d(x=seq_len, y=each_gt[:, axis], col=2, row=1, fig=fig, marker=marker_dict_gt, name="GT - {}".format(axes[axis]))
            fig = plot2d(x=seq_len, y=each_our[:, axis], col=2, row=1, fig=fig, marker=marker_dict_our, name="Ours - {}".format(axes[axis]))
            fig = plot2d(x=seq_len, y=each_cmp[:, axis], col=2, row=1, fig=fig, marker=marker_dict_cmp, name="Cmps - {}".format(axes[axis]))

          if args.selection_2d == 'diff':
            fig = plot2d(x=seq_len, y=each_our_diff[:, axis], col=2, row=1, fig=fig, marker=marker_dict_our, name="Ours - {}".format(axes[axis]))
            fig = plot2d(x=seq_len, y=each_cmp_diff[:, axis], col=2, row=1, fig=fig, marker=marker_dict_cmp, name="Cmps - {}".format(axes[axis]))
            fig = plot2d(x=seq_len, y=each_improv[:, axis], col=2, row=1, fig=fig, marker=marker_dict_gt, name="Improv - {}".format(axes[axis]))


  elif len(args.ours) > 0:
    n_sample = 50
    each_gt = gt[idx][:, [0, 1, 2]]
    # Latent 
    each_latent_gt = latent[idx][:, [0, 1]]
    each_latent_opt = latent[idx][:, [2, 3]]
    # Flag
    each_flag_gt = flag[idx][:, [0]]
    each_flag_pred = flag[idx][:, [1]]
    # if args.gt_dat == 'syn_gt':
    # latent_idx = list(np.where(np.isclose(each_flag_pred, 1.0, atol=1e-5))[0])
    # if args.gt_dat == 'gt':
    latent_idx = list(np.where(np.isclose(each_flag_gt, 1.0, atol=1e-5))[0])
    latent_idx = [0] + latent_idx

    if len(latent_idx) == 1:
      latent_idx = latent_idx
    else:
      latent_idx = latent_idx[:-1]

    each_latent_syn_gt = np.zeros(shape=(len(latent_idx), 2))

    for i, l_idx in enumerate(latent_idx):
      # direction = np.mean(each_gt[l_idx:l_idx+1+n_sample, [0, 2]], axis=0) - each_gt[l_idx, [0, 2]]
      # direction = each_gt[l_idx+1, [0, 2]] - each_gt[l_idx, [0, 2]]
      direction = np.mean(each_gt[l_idx+1:l_idx+1+n_sample, [2, 0]] - each_gt[l_idx, [2, 0]], axis=0)
      direction = direction / np.sqrt(direction[0]**2 + direction[1]**2)
      each_latent_syn_gt[i] = direction


    print("[#] Latent ")
    print(latent_idx)
    for i, latent_pos in enumerate(latent_idx):
      print(each_latent_gt[latent_pos])
      print(each_latent_opt[latent_pos])
      print(each_latent_syn_gt[i])
      # print(angle_fn(each_latent_opt[latent_pos], each_latent_gt[latent_pos]) * 180/np.pi)
      print(angle_fn(each_latent_opt[latent_pos], each_latent_syn_gt[i]) * 180/np.pi)

    if args.gt_dat == 'syn_gt':
      fig = plot_latent(latent_idx=latent_idx, latent_var=each_latent_syn_gt, data=each_gt, col=1, row=1, fig=fig, marker=marker_dict_latent_gt, name="SYN_GT_LATENT")
    else:
      fig = plot_latent(latent_idx=latent_idx, latent_var=each_latent_gt, data=each_gt, col=1, row=1, fig=fig, marker=marker_dict_latent_gt, name="GT_LATENT")

    seq_len = np.arange(each_gt.shape[0])
    for our in ours_list:
      each_our = our[idx][:, [0, 1, 2]]
      each_our_diff = np.abs(each_our - each_gt)
      fig = plot_latent(latent_idx=latent_idx, latent_var=each_latent_opt, data=each_our, col=1, row=1, fig=fig, marker=marker_dict_latent_our, name="OURS_LATENT")

      for axis in range(0, 3):
        if args.selection_2d == 'xyz':
          fig = plot2d(x=seq_len, y=each_gt[:, axis], col=2, row=1, fig=fig, marker=marker_dict_gt, name="GT - {}".format(axes[axis]))
          fig = plot2d(x=seq_len, y=each_our[:, axis], col=2, row=1, fig=fig, marker=marker_dict_our, name="Ours - {}".format(axes[axis]))

        if args.selection_2d == 'diff':
          fig = plot2d(x=seq_len, y=each_our_diff[:, axis], col=2, row=1, fig=fig, marker=marker_dict_our, name="Ours - {}".format(axes[axis]))

  else:
    print("[#] Ours and Cmps is needed")
    exit()

  fig.show()


  all_latent_gt = []
  all_latent_syn_gt = []
  all_latent_opt = []
  if len(args.ours) > 0:
    for idx in range(gt.shape[0]):
      each_gt = gt[idx][:, [0, 1, 2]]
      # Latent 
      each_latent_gt = latent[idx][:, [0, 1]]
      each_latent_opt = latent[idx][:, [2, 3]]
      # Flag
      each_flag_gt = flag[idx][:, [0]]
      each_flag_pred = flag[idx][:, [1]]
      # latent_idx = list(np.where(np.isclose(each_flag_pred, 1.0, atol=1e-5))[0])
      latent_idx = list(np.where(np.isclose(each_flag_gt, 1.0, atol=1e-5))[0])
      latent_idx = [0] + latent_idx
      if len(latent_idx) == 1:
        latent_idx = latent_idx
      elif latent_idx[-1] == each_gt.shape[0]-1:
        latent_idx = latent_idx[:-1]
      else:
        latent_idx = latent_idx

      # Latent from trajectory -> Useful for mocap
      for l_idx in latent_idx:
        direction = np.mean(each_gt[l_idx+1:l_idx+1+n_sample, [2, 0]] - each_gt[l_idx, [2, 0]], axis=0)
        # direction = each_gt[l_idx+1, [0, 2]] - each_gt[l_idx, [0, 2]]
        direction = direction / np.sqrt(direction[0]**2 +direction[1]**2)
        all_latent_syn_gt.append(np.expand_dims(direction, axis=0))

      all_latent_opt.append(each_latent_opt[latent_idx])
      all_latent_gt.append(each_latent_gt[latent_idx])


    all_latent_opt = np.concatenate(all_latent_opt, axis=0)
    all_latent_gt = np.concatenate(all_latent_gt, axis=0)
    all_latent_syn_gt = np.concatenate(all_latent_syn_gt, axis=0)

    angle = []
    for i in range(all_latent_opt.shape[0]):
      if args.gt_dat == 'syn_gt':
        angle_diff = angle_fn(all_latent_opt[i], all_latent_syn_gt[i]) * 180/np.pi
      else:
        angle_diff = angle_fn(all_latent_opt[i], all_latent_gt[i]) * 180/np.pi
      print(angle_diff)
      angle.append(angle_diff)

    print(angle_diff)
    angle = np.array(angle)

    print("[#] ANGLE")
    print("LEN : ", len(angle))
    print("MEAN ANGLE DIFFERENCE : ", np.mean(angle))
    print("SD ANGLE DIFFERENCE : ", np.std(angle))
    print("MEDIAN ANGLE DIFFERENCE : ", np.median(angle))
    threshold = 30
    max_threshold = 180 - threshold
    print("Angle +- {} : {}".format(threshold, ((np.sum(angle<threshold) + np.sum(angle>max_threshold))/len(angle))))

    max_bin = np.max(angle)
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=list(np.arange(0, 181, 20)), ec='k')
    # kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=list(np.arange(0, np.pi, 0.1)), ec='k')
    plt.hist(angle, **kwargs, label='ANGLE DIFF')
    plt.legend()
    plt.show()

