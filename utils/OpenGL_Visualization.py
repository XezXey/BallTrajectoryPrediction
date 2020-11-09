from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sys
import os
import argparse
import torch as pt
sys.path.append(os.path.realpath('../..'))
import utils.transformation as utils_transform

parser = argparse.ArgumentParser(description='Predict the 3D projectile')
parser.add_argument('--cam_params_file', dest='cam_params_file', type=str, help='Path to camera parameters file(Intrinsic/Extrinsic)')
parser.add_argument('--multiview_loss', dest='multiview_loss', help='Use multiview loss', nargs='+', default=[])
parser.add_argument('--cuda_device_num', dest='cuda_device_num', type=int, help='Provide cuda device number', default=0)
args = parser.parse_args()
utils_transform.share_args(a=args)

# GPU initialization
if pt.cuda.is_available():
  pt.cuda.set_device(args.cuda_device_num)
  device = pt.device('cuda')
  print('[%]GPU Enabled')
else:
  device = pt.device('cpu')
  print('[%]GPU Disabled, CPU Enabled')

#generate random points from -10 to 10, z-axis positive
# pos = np.random.randint(-10,10,size=(1000,3))
# pos[:,2] = np.abs(pos[:,2])

# sp = gl.GLScatterPlotItem(pos=pos)
# w.addItem(sp)

#generate a color opacity gradient
# color = np.zeros((pos.shape[0],4), dtype=np.float32)
# color[:,0] = 1
# color[:,1] = 0
# color[:,2] = 0.5
# color[0:100,3] = np.arange(0,100)/100.

def transformation(m, pts):
  m = np.linalg.inv(m.detach().cpu().numpy())
  # m = np.array([m[0, :], m[2, :], m[1, :], m[3, :]])
  # m = np.array([m[:, 0], m[:, 2], m[:, 1], m[:, 3]])
  # m[:, [1, 2]] = m[:, [2, 1]]
  # m[[1, 2], :] = m[[2, 1], :]
  # m[[1, 2], -1] = m[[2, 1], -1]
  # m[[1, 2], -1] = m[[2, 1], -1]
  m[1, -1] *= -1
  pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
  pts_transformed = pts @ m.T
  return pts_transformed[:, :-1]


# Animation the trajectory
def update():
    ## update volume colors
    global color
    color = np.roll(color,1, axis=0)
    sp.setData(color=color)

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(50)

def get_frustum_params(size):
  frustum_edges = np.array([(0, 1),
                            (0, 2),
                            (0, 3),
                            (0, 4),
                            (1, 2),
                            (1, 4),
                            (3, 2),
                            (3, 4)])

  frustum_vertices = np.array([[0, 0, 0],     # 0
                                [-2, 6, 2],     # 1
                                [2, 6, 2],      # 2 
                                [2, 6, -2],     # 3
                                [-2, 6, -2]])   # 4
  frustum_vertices = frustum_vertices * 0.2

  frustum_faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2 ,3], [1, 3, 4]])
  return frustum_edges, frustum_vertices, frustum_faces


if __name__ == '__main__':

  app = QtGui.QApplication([])
  w = gl.GLViewWidget()
  w.show()
  g = gl.GLGridItem()
  g.setSize(x=8, y=8, z=8)
  w.addItem(g)
  cam_params_dict = utils_transform.get_cam_params_dict(args.cam_params_file, device)

  frustum_edges, frustum_vertices, frustum_faces = get_frustum_params(size=0.5)
  frustum_vertices = transformation(m=cam_params_dict['main']['E'], pts=frustum_vertices)
  frustum = gl.GLMeshItem(vertexes=frustum_vertices, faces=frustum_faces, faceColors=np.array([[255,255,0,128] for i in range(frustum_faces.shape[0])]), drawEdges=True, edgeColor=(0, 0, 255, 1))
  w.addItem(frustum)

  if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
      QtGui.QApplication.instance().exec_()
