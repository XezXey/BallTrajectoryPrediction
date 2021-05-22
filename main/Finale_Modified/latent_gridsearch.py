import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()
plt.style.use('seaborn-white')
mpl.style.use('seaborn')
import argparse
import os
import math

'''
Input File convention :
1. Not in a dt-space
2. Columns sorted to : x, y, z, u, v, d, eot, latent(if exists)
3. Shape input is (n_trajectory, )
4. All sample are sorted in the same order
'''

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str, required=True)
parser.add_argument('--n_res', type=int, required=True)
parser.add_argument('--latent', type=str, default=None)
parser.add_argument('--size', type=int, default=0)
parser.add_argument('--inc', action='store_true', default=False)
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
  traj_postfix = '_all_trajectory.npy'
  loss_postfix = '_loss_landscape.npy'
  latent_postfix = '_latent.npy'
  if args.folder_path[-1] != '/':
    args.folder_path += '/'
  session_name = args.folder_path.split('/')[-2]
  traj_file = args.folder_path + session_name + traj_postfix
  loss_file = args.folder_path + session_name + loss_postfix
  latent_file = args.folder_path + session_name + latent_postfix

  traj = np.load(traj_file, allow_pickle=True)
  loss = np.load(loss_file, allow_pickle=True)


  if args.inc:
    traj = np.concatenate((traj, traj, traj))
    loss = np.concatenate((loss, loss, loss))

  # Limit the sample size
  if args.size < traj.shape[0] and args.size >=2:
    traj = traj[:args.size]
    loss = loss[:args.size]

  try:
    latent_flag = True
    latent = np.load(latent_file, allow_pickle=True)
  except FileNotFoundError:
    print("No latent to load")
    latent_flag = False

  all_latent_syn_gt = []
  degree_gt = []
  n_sample=5
  possible_angle = np.linspace(0, 360.0, int(args.n_res))
  possible_rad = possible_angle * np.pi/180
  ref_angle = [1, 0]
  # skip_idx = [6, 8, 11, 13, 24, 32, 41, 44, 51, 52, 53]
  # skip_idx = [41]
  for i in range(traj.shape[0]):
    each_gt = traj[i]
    direction = np.mean(each_gt[1:n_sample, [2, 0]] - each_gt[0, [2, 0]], axis=0)
    # direction = each_gt[100, [2, 0]] - each_gt[0, [2, 0]]
    direction = direction / np.sqrt(direction[0]**2 +direction[1]**2)
    all_latent_syn_gt.append(np.expand_dims(direction, axis=0))
    degree = 360+(np.arctan2(direction[0], direction[1])*180/np.pi)
    # degree = 360+(np.arctan2(direction[1], direction[0])*180/np.pi)
    if degree > 360:
      degree -= 360

    degree_gt.append(degree)

  degree_gt = np.array(degree_gt)
  radian_gt = degree_gt * np.pi/180

  # print("Possible angle : ", possible_angle)
  # print("GT radian : ", radian_gt)
  # print("Possible radian : ", possible_rad)
  # print("GT angle : ", degree_gt)

  '''
  Visualize the histogram of loss landscape
  '''

  all_sorted_loss = []
  q1 = []
  q2 = []
  q3 = []
  q4 = []
  pos_off = []
  neg_off = []
  if latent_flag:
    latent = latent[0][:10, [0, 1]]
    print(latent)
  print(np.sin(radian_gt[0]), np.cos(radian_gt[0]))

  for i in range(traj.shape[0]):
    re_idx = (np.abs(possible_rad - radian_gt[i])).argmin()
    # print("RE-IDX : ", re_idx)
    # print("BEFORE : ", loss[i])
    # print("FIRST : ", loss[i][re_idx:])
    # print("LAST : ", loss[i][:re_idx])
    sorted_loss = np.concatenate((loss[i][re_idx:], loss[i][:re_idx]))
    sorted_loss = sorted_loss - np.mean(sorted_loss)
    # print("AFTER : ", sorted_loss)
    # index_in_possible = (np.abs(possible_angle - degree_gt[i])).argmin()
    # print(index_in_possible)
    all_sorted_loss.append(sorted_loss)
    # Used for quadrant-seperately plot
    if degree_gt[i] >= 0 and degree_gt[i] < 90:
      q1.append(sorted_loss)
    elif degree_gt[i] >= 90 and degree_gt[i] < 180:
      q2.append(sorted_loss)
    elif degree_gt[i] >=180 and degree_gt[i] < 270:
      q3.append(sorted_loss)
    else:
      q4.append(sorted_loss)

    # Used for positive-negative degree off plot

  all_sorted_loss = np.array(all_sorted_loss)
  q1 = np.array(q1)
  q2 = np.array(q2)
  q3 = np.array(q3)
  q4 = np.array(q4)

  sum_loss = np.sum(all_sorted_loss, axis=0)
  # max_bin = np.max(sum_loss)
  sum_loss = np.concatenate((sum_loss[int(args.n_res/2):], sum_loss[0:int(args.n_res/2)]))
  # kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=list(np.arange(0, args.n_res+1, 1)), ec='k')
  kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=100, ec='k')

  # Matplotlib
  # plt.bar(x=np.arange(-int(args.n_res)//2, int(args.n_res)//2, 1), height=sum_loss, facecolor='black', edgecolor='black')
  # plt.hist(sum_loss, **kwargs)
  # plt.show()
  # exit()
  # plt.bar(x=np.arange(-int(args.n_res)//2, (int(args.n_res)//2), 1), height=sum_loss, width=1.0,  facecolor='#85b8ff', edgecolor='#599fff')
  plt.bar(x=np.arange(-int(args.n_res)//2, int(args.n_res)//2, 1), height=sum_loss, facecolor='#85b8ff', edgecolor='#85b8ff')
  plt.ylabel('Loss (subtracted by its mean)', fontsize=41)
  plt.xlabel('Azimuth angle difference (Degree)', fontsize=41)
  plt.xticks(fontsize=35)
  plt.yticks(fontsize=35)

  plt.legend()
  plt.show()


  '''
  Visualize the 4 quadrant of loss landscape seperately
  '''
  fig, axs = plt.subplots(nrows=2, ncols=2)
  fig.suptitle("Total samples = {}".format(traj.shape[0]))
  # q1
  sum_loss_q1 = np.sum(q1, axis=0)
  sum_loss_q1 = np.concatenate((sum_loss_q1[int(args.n_res/2):], sum_loss_q1[0:int(args.n_res/2)]))
  axs[0, 1].bar(x=np.arange(-int(args.n_res)//2, (int(args.n_res)//2), 1), height=sum_loss_q1, width=1.0,  facecolor='#85b8ff', edgecolor='#599fff')
  axs[0, 1].axvline(x=0, c='r')
  axs[0, 1].set_xticks(np.arange(-int(args.n_res)//2, (int(args.n_res)//2), 5))
  max_degree_q1 = max(degree_gt[np.logical_and((degree_gt >= 0), (degree_gt < 90))])
  min_degree_q1 = min(degree_gt[np.logical_and((degree_gt >= 0), (degree_gt < 90))])
  axs[0, 1].set_title('Quadrant 1 (n={}, max={:.3f}, min={:.3f})'.format(q1.shape[0], max_degree_q1, min_degree_q1))

  # q2
  sum_loss_q2 = np.sum(q2, axis=0)
  sum_loss_q2 = np.concatenate((sum_loss_q2[int(args.n_res/2):], sum_loss_q2[0:int(args.n_res/2)]))
  axs[0, 0].bar(x=np.arange(-int(args.n_res)//2, (int(args.n_res)//2), 1), height=sum_loss_q2, width=1.0,  facecolor='#85b8ff', edgecolor='#599fff')
  axs[0, 0].set_xticks(np.arange(-int(args.n_res)//2, (int(args.n_res)//2), 5))
  max_degree_q2 = max(degree_gt[np.logical_and((degree_gt >= 90), (degree_gt < 180))])
  min_degree_q2 = min(degree_gt[np.logical_and((degree_gt >= 90), (degree_gt < 180))])
  axs[0, 0].set_title('Quadrant 2 (n={}, max={:.3f}, min={:.3f})'.format(q1.shape[0], max_degree_q2, min_degree_q2))
  axs[0, 0].axvline(x=0, c='r')

  # q3
  sum_loss_q3 = np.sum(q3, axis=0)
  sum_loss_q3 = np.concatenate((sum_loss_q3[int(args.n_res/2):], sum_loss_q3[0:int(args.n_res/2)]))
  axs[1, 0].bar(x=np.arange(-int(args.n_res)//2, (int(args.n_res)//2), 1), height=sum_loss_q3, width=1.0,  facecolor='#85b8ff', edgecolor='#599fff')
  axs[1, 0].set_xticks(np.arange(-int(args.n_res)//2, (int(args.n_res)//2), 5))
  max_degree_q3 = max(degree_gt[np.logical_and((degree_gt >= 180), (degree_gt < 270))])
  min_degree_q3 = min(degree_gt[np.logical_and((degree_gt >= 180), (degree_gt < 270))])
  axs[1, 0].set_title('Quadrant 3 (n={}, max={:.3f}, min={:.3f})'.format(q3.shape[0], max_degree_q3, min_degree_q3))
  axs[1, 0].axvline(x=0, c='r')

  # q4
  sum_loss_q4 = np.sum(q4, axis=0)
  sum_loss_q4 = np.concatenate((sum_loss_q4[int(args.n_res/2):], sum_loss_q4[0:int(args.n_res/2)]))
  axs[1, 1].bar(x=np.arange(-int(args.n_res)//2, (int(args.n_res)//2), 1), height=sum_loss_q4, width=1.0,  facecolor='#85b8ff', edgecolor='#599fff')
  axs[1, 1].set_xticks(np.arange(-int(args.n_res)//2, (int(args.n_res)//2), 5))
  max_degree_q4 = max(degree_gt[np.logical_and((degree_gt >= 270), (degree_gt < 360))])
  min_degree_q4 = min(degree_gt[np.logical_and((degree_gt >= 270), (degree_gt < 360))])
  axs[1, 1].set_title('Quadrant 4 (n={}, max={:.3f}, min={:.3f})'.format(q4.shape[0], max_degree_q4, min_degree_q4))
  axs[1, 1].axvline(x=0, c='r')

  plt.show()
  exit()

  '''
  Visualize the histogram of angle error by remove the trjectory that latent didn't help that much.
  '''
  if args.latent is not None:
    latent_ = np.load(args.latent, allow_pickle=True)

  poss_sin = np.expand_dims(np.sin(possible_rad), axis=-1)
  poss_cos = np.expand_dims(np.cos(possible_rad), axis=-1)
  pred_dir = np.concatenate((poss_sin, poss_cos), axis=-1)
  print(pred_dir.shape)
  print(np.std(loss))
  all_angle_error = []
  # for n in np.arange(0, 1, 0.1):
  n = 0.1
  threshold = 0.005
  for i in range(traj.shape[0]):
    each_loss = loss[i]
    each_std = np.std(loss[i])
    each_mean = np.mean(loss[i])
    # conditioning = (each_std-n)
    if each_std < threshold:
      print("DROP")
      continue
    # conditioned = np.where(each_loss < conditioning)[0]
    # for j in conditioned:
    else:
      # for j in range(each_loss.shape[0]):
        # angle_error = angle_fn(degree_gt[i] - possible_angle[j]
        # angle_error = angle_fn(all_latent_syn_gt[i], pred_dir[j])
      angle_error = angle_fn(all_latent_syn_gt[i], latent_[i][0, [2, 3]])
      all_angle_error.append(angle_error * 180/np.pi)

  # kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=list(np.arange(0, args.n_res+1, 1)), ec='k')
  kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=list(np.arange(0, 180, 10)), ec='k')
  # print(all_angle_error[all_angle_error>180])
  plt.hist(all_angle_error, **kwargs, label='Threshold = {:.2f}'.format(threshold))
  plt.xlabel('Angle difference (Degree)')
  plt.ylabel('Number of trajectories / sequences')
  plt.legend()
  plt.show()
  folder = './etc/latent/'
  if not os.path.exists(folder):
    os.makedirs(folder)
  plt.savefig(fname=folder + 'std-{:.2f}.png'.format(threshold))
  plt.cla()
    # plt.axhline(y=each_std)
    # plt.plot(each_loss_below_std.reshape(-1))
    # plt.plot(np.array(each_loss).reshape(-1))
    # exit()

  print(all_sorted_loss)
  print(all_latent_syn_gt)
  exit()

