import numpy as np
import matplotlib.pyplot as plt
import plotly
import argparse
import glob
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', dest='data', type=str, required=True)
parser.add_argument('--rmv_undetected_wall', default = False, action='store_true')
args = parser.parse_args()

marker_dict_latent = dict(color='rgba(0, 255, 0, 0.4)', size=4)

def vis(i, data):
  trajectory = data[i]
  trajectory_temp = np.cumsum(data[i], axis=0)
  x, y, z = trajectory_temp[:, 0], trajectory_temp[:, 1], trajectory_temp[:, 2]
  fig = make_subplots(rows=1, cols=1, specs=[[{'type':'scatter3d'}]])
  fig['layout']['scene1'].update(yaxis=dict(nticks=5, range=[-1, 5]))

  # Trajectory
  fig.add_trace(go.Scatter3d(x=x, y=y, z=z))

  # Latent - sis, cosine
  n_latent = 10
  for i in range(n_latent):
    idx = np.random.randint(low=0, high=trajectory.shape[0])
    sin, cos = trajectory[:, 9], trajectory[:, 10]
    print(sin[0], cos[0], sin[0]**2 + cos[0]**2)
    latent_x = np.array([x[idx], x[idx] + cos[idx]])
    latent_y = np.array([y[idx], y[idx]])
    latent_z = np.array([z[idx], z[idx] + sin[idx]])
    fig.add_trace(go.Scatter3d(x=latent_x, y=latent_y, z=latent_z, marker=marker_dict_latent, line=dict(width=10)))
  fig.show()

def one_sample(data):

  # Visualize 1 cases
  idx = np.random.randint(low=0, high=data.shape[0])
  vis(i=idx, data=data)

  trajectory = data[idx]
  trajectory_temp = np.cumsum(data[idx], axis=0)
  x, y, z = trajectory_temp[:, 0], trajectory_temp[:, 1], trajectory_temp[:, 2]
  sin, cos = trajectory[:, 9], trajectory[:, 10]
  # Check the value of sin-cosine
  dx = x[1:] - x[:-1]
  dz = z[1:] - z[:-1]
  dx_norm = dx / np.sqrt((dx**2 + dz**2))
  dz_norm = dz / np.sqrt((dx**2 + dz**2))
  zero = np.zeros(shape=dx_norm.shape)

  tol = 1e-3

  '''
  print("*"*50)
  print("[#] COMPARE SIN = {} with dz_norm = {}".format(sin[0], np.mean(dz_norm)))
  print("===> MEAN SIN : {}, STD SIN : {}, UNIQUE : {}".format(np.mean(sin), np.std(sin), np.unique(sin)))
  print("===> DIFF : ", np.abs(np.mean(sin) - np.mean(dz_norm)))
  print("===> SIN Equality : ", np.all(np.abs(sin[1:] - np.mean(dz_norm)) < tol))
  print("*"*50)
  print("COMPARE COS = {} with dx_norm = {}".format(cos[0], np.mean(dx_norm)))
  print("===> MEAN COS : {}, STD COS : {}, UNIQUE : {}".format(np.mean(cos), np.std(cos), np.unique(cos)))
  print("===> DIFF : ", np.abs(np.mean(cos) - np.mean(dx_norm)))
  print("===> COS Equality : ", np.all(np.abs(cos[1:] - np.mean(dx_norm)) < tol))
  '''

def latent_equality(data):
  tol = 5e-2
  fail = 0
  fail_count = 0
  fail_index = []
  all_degree_diff = []
  deg_count = 0
  deg_fail = []
  sin_fail = {}
  cos_fail = {}
  for i in tqdm.tqdm(range(data.shape[0])):
    # Trajectory
    trajectory = data[i]
    trajectory_temp = np.cumsum(data[i], axis=0)
    x, y, z = trajectory_temp[:, 0], trajectory_temp[:, 1], trajectory_temp[:, 2]
    # Delta
    dx = x[1:] - x[:-1]
    dz = z[1:] - z[:-1]

    # print(x)
    # print(z)
    # print(dx)
    # print(dz)
    dx_norm = dx / (np.sqrt((dx**2 + dz**2)) + 1e-6)
    dz_norm = dz / (np.sqrt((dx**2 + dz**2)) + 1e-6)
    # dx_norm = dx / np.sqrt((dx**2 + dz**2))
    # dz_norm = dz / np.sqrt((dx**2 + dz**2))
    # print(dx_norm)
    # print(dz_norm)
    # Latent
    sin, cos = trajectory[:, 9], trajectory[:, 10]
    # Degree
    gt_deg = np.arctan2(np.mean(sin), np.mean(cos)) * 180.0 / np.pi
    traj_deg = np.arctan2(np.mean(dz_norm), np.mean(dx_norm)) * 180.0 / np.pi
    degree_diff = np.abs(gt_deg - traj_deg)
    all_degree_diff.append(degree_diff)
    if (not np.all(np.abs(cos[1:] - np.mean(dx_norm)) < tol)) or (not np.all(np.abs(sin[1:] - np.mean(dz_norm)) < tol)):
      fail = 1
      '''
      print("#"*50)
      print("#"*50)
      print("*"*50)
      print("[#] COMPARE SIN = {} with dz_norm = {}".format(sin[0], np.mean(dz_norm)))
      print("===> MEAN SIN : {}, STD SIN : {}, UNIQUE : {}".format(np.mean(sin), np.std(sin), np.unique(sin)))
      print("===> DIFF : ", np.abs(np.mean(sin) - np.mean(dz_norm)))
      print("===> SIN Equality : ", np.all(np.abs(sin[1:] - np.mean(dz_norm)) < tol))
      print("*"*50)
      print("[#] COMPARE COS = {} with dx_norm = {}".format(cos[0], np.mean(dx_norm)))
      print("===> MEAN COS : {}, STD COS : {}, UNIQUE : {}".format(np.mean(cos), np.std(cos), np.unique(cos)))
      print("===> DIFF : ", np.abs(np.mean(cos) - np.mean(dx_norm)))
      print("===> COS Equality : ", np.all(np.abs(cos[1:] - np.mean(dx_norm)) < tol))
      print("*"*50)
      print("[#] DEGREE DIFF = ", degree_diff)
      # vis(i=i, data=data)
      # input()
      '''
      fail_count+=1
      sin_fail[i] = [np.unique(sin)[0], np.mean(dz_norm)]
      cos_fail[i] = [np.unique(cos)[0], np.mean(dx_norm)]
      if degree_diff > 1:
        deg_count+=1
        deg_fail.append(degree_diff)
        fail_index.append(i)
        # vis(i=i, data=data)
        # input()

  if fail:
    print("[#] Equality check is failed")
    print("===> N = : ", fail_count)
    print("[#] Degree diff > 1.0")
    print("===> N = : ", deg_count)
    if deg_count > 0:
      print("===> Max diff = {}, Min diff = {}".format(np.max(deg_fail), np.min(deg_fail)))
  else:
    print("[#] Latent is correct")

  print(np.max(all_degree_diff))
  n_bins = np.linspace(start=0.0, stop=np.max(all_degree_diff), num=10)
  kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=n_bins , ec='k')
  plt.hist(all_degree_diff, **kwargs, label='Degree diff (mean={:.3f}, std={:.3f})'.format(np.mean(all_degree_diff), np.std(all_degree_diff)))
  plt.show()
  if np.isnan(np.mean(all_degree_diff)):
    print("Nan raised : Some x-z are in the same position for t and t-1")
    exit()

  return fail_index

def check_xz_distrubution(data):
  all_x = []
  all_z = []
  for i in tqdm.tqdm(range(data.shape[0]), desc="Check a xz-distribution"):
    # Trajectory
    trajectory = data[i]
    trajectory_temp = np.cumsum(data[i], axis=0)
    x, y, z = trajectory_temp[:, 0], trajectory_temp[:, 1], -trajectory_temp[:, 2]  # Just invert for make the view same as unity
    all_x.append(x)
    all_z.append(z)

  all_x = np.concatenate(all_x)
  all_z	 = np.concatenate(all_z)


  n_bins = 300
  kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=n_bins , ec='k')
  # plt.hist(all_x, **kwargs, label='X (mean={:.3f}, std={:.3f})'.format(np.mean(all_x), np.std(all_x)))
  # plt.hist(all_z, **kwargs, label='Z (mean={:.3f}, std={:.3f})'.format(np.mean(all_z), np.std(all_z)))

  fig = plt.figure(figsize=(8, 8))
  plt.suptitle("Distribution of xz-plane")
  left, width = 0.1, 0.65
  bottom, height = 0.1, 0.65
  spacing = 0.01


  rect_hist2d = [left, bottom, width, height]
  rect_histx = [left, bottom + height + spacing, width, 0.2]
  rect_histy = [left + width + spacing, bottom, 0.2, height]

  ax = fig.add_axes(rect_hist2d)
  ax_histx = fig.add_axes(rect_histx, sharex=ax)
  ax_histx.hist(all_x, **kwargs, label='X (mean={:.3f}, std={:.3f})'.format(np.mean(all_x), np.std(all_x)))
  ax_histx.axvline(x=0, c='r', alpha=0.3)

  ax_histz = fig.add_axes(rect_histy, sharey=ax)
  ax_histz.hist(all_z, **kwargs, label='Z (mean={:.3f}, std={:.3f})'.format(np.mean(all_z), np.std(all_z)), orientation='horizontal')
  ax_histz.axhline(y=0, c='r', alpha=0.3)

  ax.hist2d(x=all_x, y=all_z, bins=n_bins, density=True)
  ax.axvline(x=0, c='r', alpha=0.7)
  ax.axhline(y=0, c='r', alpha=0.7)

  ax.axvline(x=np.mean(all_x), c='b', alpha=0.7)
  ax.axhline(y=np.mean(all_z), c='b', alpha=0.7)

  ax_histx.legend()
  ax_histz.legend()
  plt.show()

def check_latent_distrubution(data):
  all_sin = []
  all_cos = []
  for i in tqdm.tqdm(range(data.shape[0]), desc='Check a latent distribution'):
    trajectory = data[i]
    sin, sin_idx = np.unique(trajectory[:, 9], return_index=True)
    cos, cos_idx = np.unique(trajectory[:, 10], return_index=True)
    sin = sin[sin_idx.argsort()]
    cos = cos[cos_idx.argsort()]

    # sin, cos = trajectory[0, 9], trajectory[0, 10]
    all_sin.append(sin)
    all_cos.append(cos)

  # all_sin = np.array(all_sin)
  # all_cos = np.array(all_cos)
  all_sin = np.concatenate(all_sin)
  all_cos = np.concatenate(all_cos)
  # print("Correction of sin/cos : ", np.where(np.sqrt(all_sin**2 + all_cos**2) < 0.95))
  # idx = np.where(np.sqrt(all_sin**2 + all_cos**2) < 0.95)[0]
  # vis(data=data, i=idx[0])
  # exit()

  deg = np.arctan2(all_sin, all_cos) * 180/np.pi
  deg[deg < 0] = deg[deg < 0] + 360
  rad = deg * np.pi/180.0

  # Plotting in polar coordinate
  kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, ec='k')

  n_bins = 60
  sub_div = n_bins
  bins = np.linspace(start=0.0, stop=2*np.pi, num=sub_div+1)
  n, _, _ = plt.hist(rad, bins, **kwargs)
  plt.clf()
  width = 2 * np.pi / sub_div
  ax = plt.subplot(1, 1, 1, projection='polar')
  bars = ax.bar(bins[:sub_div], n, width=width, bottom=0.0, label='Rad (mean={:.3f}, std={:.3f})'.format(np.mean(rad), np.std(rad)))
  for bar in bars:
    bar.set_alpha(0.5)

  plt.legend()
  c_hist_fn = './etc/c_hist.png'
  plt.savefig(c_hist_fn)
  plt.show()
  plt.clf()

  # Plotting in standard histogram
  kwargs = dict(histtype='stepfilled', alpha=0.3, density=False, bins=n_bins , ec='k')
  plt.hist(x=deg, **kwargs, label='Deg (mean={:.3f}, std={:.3f})'.format(np.mean(deg), np.std(deg)))

  plt.legend()
  b_hist_fn = './etc/b_hist.png'
  plt.savefig(b_hist_fn)
  plt.show()
  plt.clf()

  c_hist = plt.imread(c_hist_fn)
  b_hist = plt.imread(b_hist_fn)

  plt.imshow(np.hstack((b_hist, c_hist)))
  plt.show()


if __name__ == '__main__':
  print("[#] Latent checking")
  data = np.load(args.data, allow_pickle=True)
  print("===> Shape : ", data[0].shape)
  one_sample(data)
  fail_index = latent_equality(data)
  check_xz_distrubution(data)
  check_latent_distrubution(data)

  if args.rmv_undetected_wall and len(fail_index) > 0:
    data_ = np.delete(data, fail_index)
    print("#"*100)
    print("[#] Before remove the undetected wall collision")
    print("===>Fail : ", fail_index)
    print("===>Init shape : ", data.shape)
    print("[#] After remove the undetected wall collision")
    fail_index = latent_equality(data_)
    print("===>Fail : ", fail_index)
    print("===>Final shape : ", data_.shape)
    print("#"*100)
    np.save(file='{}_pre.npy'.format(args.data.split('.')[0]), arr=data_)
