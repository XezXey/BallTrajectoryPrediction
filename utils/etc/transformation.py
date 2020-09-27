import argparse
import numpy
import matplotlib.pyplot as plt
import json
import numpy as np

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--cam_params_file', dest='cam_params_file', type=str, help='Path to camera parameters file(Intrinsic/Extrinsic)')
  args = parser.parse_args()

  # Load camera parameters : ProjectionMatrix and WorldToCameraMatrix
  with open(args.cam_params_file) as cam_params_json:
    cam_params_file = json.load(cam_params_json)
    cam_params = dict({'projectionMatrix':cam_params_file['mainCameraParams']['projectionMatrix'], 'worldToCameraMatrix':cam_params_file['mainCameraParams']['worldToCameraMatrix'], 'width':cam_params_file['mainCameraParams']['width'], 'height':cam_params_file['mainCameraParams']['height']})
  projection_matrix = np.array(cam_params['projectionMatrix']).reshape(4, 4)
  projection_matrix = np.array([projection_matrix[0, :], projection_matrix[1, :], projection_matrix[3, :], [0, 0, 0, 1]], dtype=np.float32)
  E_inv = np.linalg.inv(np.array(cam_params['worldToCameraMatrix']).reshape(4, 4))
  width = cam_params['width']
  height = cam_params['height']

  print("ProjectionMatrix = \n", projection_matrix)
  print("E_inv : \n", E_inv)

  world = np.array([[2.5, 0, -6, 1]])

  depth = np.array([[59.4]])
  screen = np.array([[843.3, 523.7, 1, 1]])

  ndc = screen.copy()
  ndc[:, 0] /= width
  ndc[:, 1] /= height
  ndc[:, :2] = (ndc[:, :2] * 2)-1
  ndc[:, :-1] *= 1
  projection_matrix_inv = np.linalg.inv(projection_matrix)

  camera = ndc @ projection_matrix_inv.T
  world = camera @ E_inv.T
  print("Screen : ", screen)
  print("NDC : ", ndc)
  print("Camera : ", camera)
  print("World : ", world)
