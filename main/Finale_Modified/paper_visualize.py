import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import argparse

'''
Input File convention :
1. Not in a dt-space
2. Columns sorted to : x, y, z, u, v, d, eot, latent(if exists)
3. Shape input is (n_trajectory, )
4. All sample are sorted in the same order
'''

parser = argparse.ArgumentParser()
parser.add_argument('--ours', nargs='+', required=True, default=[])
parser.add_argument('--gt', type=str, required=True, default=None)
parser.add_argument('--cmps', nargs='+', default=[])
parser.add_argument('--sample_idx', type=int, required=False, default=None)
parser.add_argument('--selection_2d', type=str, default='xyz')
args = parser.parse_args()

def plot3d(data, col, row, fig, marker, name):
  x = data[:, 0]
  y = data[:, 1]
  z = data[:, 2]
  fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=marker, mode='markers', name=name), row=row, col=col)
  return fig

def plot2d(x, y, col, row, fig, marker, name):
  fig.add_trace(go.Scatter(x=x, y=y, marker=marker, mode='markers+lines', name=name), row=row, col=col)
  return fig

if __name__ == '__main__':
  ours_list = []
  cmps_list = []

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

  # PLOTTING 
  fig = make_subplots(rows=1, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter'}]])
  marker_dict_gt = dict(color='rgba(0, 0, 255, 1.0)', size=3)
  marker_dict_our = dict(color='rgba(255, 0, 0, 0.7)', size=3)
  marker_dict_cmp = dict(color='rgba(0, 0, 0, 0.7)', size=3)

  marker_dict_2d = dict(color='rgba(255, 0, 0, 0.7)', size=3)

  # 3D plot
  if args.sample_idx is not None:
    idx = [args.sample_idx]
  else:
    idx = np.random.randint(0, ours_list[0].shape)

  # Plot ours
  if len(args.ours) > 0:
    for our in ours_list:
      fig = plot3d(data=our[idx][0], col=1, row=1, fig=fig, marker=marker_dict_our, name="Ours")

  # Plot cmps
  if len(args.cmps) > 0:
    for cmp in cmps_list:
      fig = plot3d(data=cmp[idx][0], col=1, row=1, fig=fig, marker=marker_dict_cmp, name="Cmps")

  # Plot gt
  fig = plot3d(data=gt[idx][0], col=1, row=1, fig=fig, marker=marker_dict_gt, name="GT")

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


  else:
    print("[#] Ours and Cmps is needed")
    exit()
  fig.show()

