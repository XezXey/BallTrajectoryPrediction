import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tqdm
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
  if args.folder_path[-1] != '/':
    args.folder_path += '/'
  session_name = args.folder_path.split('/')[-2]
  traj_file = args.folder_path + session_name + traj_postfix
  loss_file = args.folder_path + session_name + loss_postfix

  traj = np.load(traj_file, allow_pickle=True)
  loss = np.load(loss_file, allow_pickle=True)[:, :-1]
  print(loss.shape)


  all_latent_syn_gt = []
  degree_gt = []
  n_sample=5
  possible_angle = np.linspace(0, 360.0, int(args.n_res))[:-1]
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
    if degree > 360:
      degree -= 360

    degree_gt.append(degree)

  degree_gt = np.array(degree_gt)
  radian_gt = degree_gt * np.pi/180

  # plt.hist(degree_gt)
  # plt.show()

  # print("Possible angle : ", possible_angle)
  # print("GT radian : ", radian_gt)
  # print("Possible radian : ", possible_rad)
  # print("GT angle : ", degree_gt)

  all_sorted_loss = []
  for i in range(traj.shape[0]):
    re_idx = (np.abs(possible_rad - radian_gt[i])).argmin()
    # print(possible_rad - radian_gt[i])
    # print(possible_rad)
    # print(radian_gt[i])
    # exit()
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

  all_sorted_loss = np.array(all_sorted_loss)
  # sum_loss = np.sum(all_sorted_loss, axis=0)
  sum_loss = np.mean(all_sorted_loss, axis=0)
  # max_bin = np.max(sum_loss)
  sum_loss = np.concatenate((sum_loss[int(args.n_res/2):], sum_loss[0:int(args.n_res/2)]))
  # kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=list(np.arange(0, args.n_res+1, 1)), ec='k')
  # plt.hist(sum_loss, **kwargs, label='ANGLE DIFF')

  '''
  # Plotly
  plt.cla()
  layout = go.Layout(bargap=0, bargroupgap=0)
  fig = go.Figure(data = [
    go.Bar(name='Loss landscape', x=np.arange(-int(args.n_res)//2, int(args.n_res)//2, 1), y=sum_loss)],
                  layout=layout)
  fig.show()
  '''

  '''
  # Seaborn
  plt.cla()
  ax = sns.barplot(x=np.arange(-int(args.n_res)//2, int(args.n_res)//2, 1), y=sum_loss, facecolor='black', edgecolor='black')
  # ax.set(xticks=np.arange(-int(args.n_res)//2, int(args.n_res)//2, 20))
  plt.show()
  exit()
  '''

  # Matplotlib
  # plt.bar(x=np.arange(-int(args.n_res)//2, int(args.n_res)//2, 1), height=sum_loss, facecolor='black', edgecolor='black')
  # plt.bar(x=np.arange(-int(args.n_res)//2, (int(args.n_res)//2)-1, 1), height=sum_loss, width=1.0),  facecolor='#85b8ff', edgecolor='#599fff')
  color = '#4c72b0'
  plt.bar(x=np.arange(-int(args.n_res)//2, (int(args.n_res)//2)-1, 1), height=sum_loss, width=1.0, alpha=0.3)#, facecolor=color, edgecolor=color, alpha=0.5)
  # plt.bar(x=np.arange(-int(args.n_res)//2, int(args.n_res)//2, 1), height=sum_loss, facecolor='#85b8ff', edgecolor='#85b8ff')
  plt.ylabel('Average normalized loss', fontsize=41)
  plt.xlabel('Azimuth angle difference (Degree)', fontsize=41)
  plt.xticks(fontsize=35)
  plt.yticks(fontsize=35)

  plt.legend()
  plt.show()
  # exit()

  '''
  if args.latent is not None:
    latent_ = np.load(args.latent, allow_pickle=True)

  poss_sin = np.expand_dims(np.sin(possible_rad), axis=-1)
  poss_cos = np.expand_dims(np.cos(possible_rad), axis=-1)
  pred_dir = np.concatenate((poss_sin, poss_cos), axis=-1)
  # skip_idx = [6, 8, 11, 13, 24, 32, 41, 44, 51, 52, 53]   # for mocap
  for threshold in tqdm.tqdm(np.arange(0, 2, 0.25)):
    # threshold = 0.005
    all_angle_error = []
    for i in range(traj.shape[0]):
      # if i in skip_idx:
        # print("SKIP")
        # continue
      each_loss = loss[i]
      each_std = np.std(loss[i])
      each_mean = np.mean(loss[i])
      if each_std < threshold:
        # print("DROP")
        continue
      # conditioned = np.where(each_loss < conditioning)[0]
      # for j in conditioned:
      else:
        # Use the optimized one
        angle_error = angle_fn(all_latent_syn_gt[i], latent_[i][0, [2, 3]])
        # Use a min(loss)
        # min_loss_latent = possible_rad[np.argmin(each_loss)]
        # each_min_latent = [np.sin(min_loss_latent), np.cos(min_loss_latent)]
        # angle_error = angle_fn(all_latent_syn_gt[i], each_min_latent)
        # print(angle_error)
        # exit()
        all_angle_error.append(angle_error * 180/np.pi)

    # kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=list(np.arange(0, args.n_res+1, 1)), ec='k')
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=list(np.arange(0, 180, 5)), ec='k')
    plt.hist(all_angle_error, **kwargs, label='Threshold = {:.3f}'.format(threshold))
    # plt.yticks(np.arange(0, 150, 15))
    plt.xlabel('Angle difference (Degree)', fontsize=37)
    plt.ylabel('Number of trajectories / sequences', fontsize=37)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.legend(fontsize=25)
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

  # print(all_sorted_loss)
  # print(all_latent_syn_gt)

  '''
