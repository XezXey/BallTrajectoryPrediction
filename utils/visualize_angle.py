import numpy as np
import matplotlib.pyplot as plt
import plotly
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
parser.add_argument('--latent', nargs='+')
parser.add_argument('--i', type=int, default=None)
parser.add_argument('--selected_features', dest='selected_features', help='Specify the selected features columns(eot, og, ', nargs='+', default=[])
args = parser.parse_args()

# Get selected features to input into a network
features = ['x', 'y', 'z', 'u', 'v', 'd', 'eot', 'og', 'rad', 'f_sin', 'f_cos', 'fx', 'fy', 'fz', 'fx_norm', 'fy_norm', 'fz_norm', 'g']
x, y, z, u, v, d, eot, og, rad, f_sin, f_cos, fx, fy, fz, fx_norm, fy_norm, fz_norm, g = range(len(features))

if __name__ == '__main__':
  data = np.load(args.input_file, allow_pickle=True)
  if args.i is None:
    i = np.random.randint(low=0, high=data.shape[0], size=(1))[0]
  else:
    i = args.i
  trajectory = data[i]
  lengths = [trajectory.shape[0] for trajectory in data]

  trajectory[:, [x, y, z]] = np.cumsum(trajectory[:, [x, y, z]], axis=0)
  flag = trajectory[:, [eot]]
  flag = np.concatenate((np.zeros((1, 1)), flag), axis=0)
  close = np.isclose(flag, 1., atol=5e-1)
  where = np.where(close == True)[0]
  where = where[where < lengths[i]]
  # print(where)
  if len(where) == 0:
    where = [0]
  else:
    where = [0] + list(where-1)
  print(where)

  fig = make_subplots(rows=1, cols=1, specs=[[{'type':'scatter3d'}]])
  fig.add_trace(go.Scatter3d(x=trajectory[:, x], y=trajectory[:, y], z=trajectory[:, z]))

  # Reference Axis
  for i in range(4):
    factor = [False, False, False]
    if i < 3:
      factor[i] = True
    else:
      break
    ref_x = [0.0, 1.0 * factor[0]]
    ref_y = [0.0, 1.0 * factor[1]]
    ref_z = [0.0, 1.0 * factor[2]]
    fig.add_trace(go.Scatter3d(x=ref_x, y=ref_y, z=ref_z, mode='lines', line=dict(width=10), name='F'))

  # Latent
  for pos in where:
    print("Rad : ", trajectory[pos, rad])
    print("Degree : ", trajectory[pos, rad]*180/np.pi)
    print("Rad to Angle(cos, sin) : ({}, {})".format(np.sin(trajectory[pos, rad]), np.cos(trajectory[pos, rad])))
    if 'angle' in args.latent:
      print("Angle (cos, sin) : ({}, {})".format(trajectory[pos, f_cos], trajectory[pos, f_sin]))
      arrow_x = [trajectory[pos, x], trajectory[pos, x] + trajectory[pos, f_cos]]
      arrow_y = [trajectory[pos, y], trajectory[pos, y]]
      arrow_z = [trajectory[pos, z], trajectory[pos, z] + trajectory[pos, f_sin]]
      fig.add_trace(go.Scatter3d(x=arrow_x, y=arrow_y, z=arrow_z, mode='lines', line=dict(width=10), name='Angle'))
    if 'force' in args.latent:
      print("Force (fx, fy, fz) : ({}, {}, {})".format(trajectory[pos, fx], trajectory[pos, fy], trajectory[pos, fz]))
      for i in range(4):
        factor = [False, False, False]
        if i < 3:
          factor[i] = True
        else:
          factor = [True, True, True]
        arrow_x = [trajectory[pos, x], trajectory[pos, x] + trajectory[pos, fx] * factor[0]]
        arrow_y = [trajectory[pos, y], trajectory[pos, y] + trajectory[pos, fy] * factor[1]]
        arrow_z = [trajectory[pos, z], trajectory[pos, z] + trajectory[pos, fz] * factor[2]]
        fig.add_trace(go.Scatter3d(x=arrow_x, y=arrow_y, z=arrow_z, mode='lines', line=dict(width=10), name='F'))
    if 'force_norm' in args.latent:
      print("Force norm (fx_norm, fy_norm, fz_norm) : ({}, {}, {})".format(trajectory[pos, fx_norm], trajectory[pos, fy_norm], trajectory[pos, fz_norm]))
      for i in range(4):
        factor = [False, False, False]
        if i < 3:
          factor[i] = True
        else:
          factor = [True, True, True]
        arrow_x = [trajectory[pos, x], trajectory[pos, x] + trajectory[pos, fx_norm] * factor[0]]
        arrow_y = [trajectory[pos, y], trajectory[pos, y] + trajectory[pos, fy_norm] * factor[1]]
        arrow_z = [trajectory[pos, z], trajectory[pos, z] + trajectory[pos, fz_norm] * factor[2]]
        fig.add_trace(go.Scatter3d(x=arrow_x, y=arrow_y, z=arrow_z, mode='lines', line=dict(width=10), name='F_norm'))



  fig.show()
