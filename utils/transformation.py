import torch as pt
import json
import numpy as np

def projectToWorldSpace(uv, depth, cam_params_dict, device):
  # print(uv.shape, depth.shape)
  depth = depth.view(-1)
  screen_width = cam_params_dict['width']
  screen_height = cam_params_dict['height']
  I_inv = cam_params_dict['I_inv']
  E_inv = cam_params_dict['E_inv']
  uv = pt.div(uv, pt.tensor([screen_width, screen_height]).to(device)) # Normalize : (width, height) -> (-1, 1)
  uv = (uv * 2.0) - pt.ones(size=(uv.size()), dtype=pt.float32).to(device) # Normalize : (width, height) -> (-1, 1)
  uv = (uv.t() * depth).t()   # Normalize : (-1, 1) -> (-depth, depth) : Camera space (x', y', d, 1)
  uv = pt.stack((uv[:, 0], uv[:, 1], depth, pt.ones(depth.shape[0], dtype=pt.float32).to(device)), axis=1) # Stack the screen with depth and w ===> (x, y, depth, 1)
  uv = ((E_inv @ I_inv) @ uv.t()).t() # Reprojected
  return uv[:, :3]

def get_cam_params_dict(cam_params_file, device):
  '''
  Return the cameras parameters use in reconstruction
  '''
  cam_params_dict = {}
  with open(cam_params_file) as cam_params_json:
    cam_params_file = json.load(cam_params_json)
    cam_params = dict({'projectionMatrix':cam_params_file['mainCameraParams']['projectionMatrix'], 'worldToCameraMatrix':cam_params_file['mainCameraParams']['worldToCameraMatrix'], 'width':cam_params_file['mainCameraParams']['width'], 'height':cam_params_file['mainCameraParams']['height']})
  projection_matrix = np.array(cam_params['projectionMatrix']).reshape(4, 4)
  projection_matrix = pt.tensor([projection_matrix[0, :], projection_matrix[1, :], projection_matrix[3, :], [0, 0, 0, 1]], dtype=pt.float32)
  cam_params_dict['I'] = projection_matrix
  cam_params_dict['I_inv'] = pt.inverse(projection_matrix).to(device)

  cam_params_dict['E'] = pt.tensor(cam_params['worldToCameraMatrix']).view(4, 4).to(device)
  cam_params_dict['E_inv'] = pt.inverse(pt.tensor(cam_params['worldToCameraMatrix']).view(4, 4)).to(device)
  cam_params_dict['width'] = cam_params['width']
  cam_params_dict['height'] = cam_params['height']

  return cam_params_dict


