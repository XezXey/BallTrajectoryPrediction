import torch as pt
import matplotlib.pyplot as plt
import numpy as np
import math

# TrajectoryOptimization Class
class OptimizationLatentAnalyze(pt.nn.Module):
  def __init__(self, pred_xyz, cam_params_dict, gt_dict, latent_size, model_dict, n_refinement, pred_dict, latent_code):
    super(OptimizationLatentAnalyze, self).__init__()
    self.pred_xyz = pred_xyz
    self.cam_params_dict = cam_params_dict
    self.gt_dict = gt_dict
    self.model_dict = model_dict
    self.n_refinement = n_refinement
    self.pred_dict = pred_dict
    self.latent_code = latent_code

  def forward(self, in_f, lengths):
    for idx in range(self.n_refinement):
      pred_refinement, (_, _) = self.model_dict['model_refinement_{}'.format(idx)](in_f=in_f, lengths=lengths)
    return pred_refinement

  def construct_latent(self):
    # This use EOT to split the latent. Stack and repeat to have the same shape with the "pred_xyz" 
    flag = self.pred_dict['model_flag']
    lengths = self.gt_dict['lengths']
    flag = pt.cat((pt.zeros((flag.shape[0], 1, 1)).cuda(), flag), dim=1)
    close = pt.isclose(flag, pt.tensor(1.), atol=5e-1)
    latent_list = []
    for i in range(flag.shape[0]):
      where = pt.where(close[i] == True)[0]
      where = where[where < lengths[i]]
      if len(where) == 0:
        # Flag prediction goes wrong
        # latent = pt.nn.Parameter(pt.rand(1, self.latent_size, dtype=pt.float32).cuda() * 100., requires_grad=True).cuda()
        self.latent.append(pt.nn.Parameter(pt.rand(1, self.latent_size, dtype=pt.float32).cuda(), requires_grad=True).cuda())
      else:
        # Flag prediction work properly
        n_latent = len(where)
        # latent = pt.nn.Parameter(pt.rand(n_latent+1, self.latent_size, dtype=pt.float32).cuda() * 100., requires_grad=True).cuda()
        self.latent.append(pt.nn.Parameter(pt.rand(n_latent+1, self.latent_size, dtype=pt.float32).cuda(), requires_grad=True).cuda())

    return latent_list

  def manipulate_latent(self, latent):
    remaining_size = self.latent_size
    latent_pointer = 0
    latent_in = []
    if 'angle' in self.latent_code and remaining_size > 0:
      #####################################
      ############### ANGLE ###############
      #####################################
      latent_angle = pt.cat((pt.sin(latent[..., [latent_pointer]] * math.pi/180.0), pt.cos(latent[..., [latent_pointer]] * math.pi/180.0)), dim=2)
      # latent_angle = pt.cat((pt.sin(latent[..., [latent_pointer]]), pt.cos(latent[..., [latent_pointer]])), dim=2)
      latent_in.append(latent_angle)
      print("Manipulate Angle : ", latent_angle.shape)
      remaining_size -= 1
      latent_pointer += 1

    if 'sin_cos' in self.latent_code and remaining_size > 0:
      # latent_sin_cos = pt.cat((latent[..., [latent_pointer:latent_pointer+2]], latent[..., [latent_pointer:latent_pointer+2]])), dim=2)
      latent_sin_cos = latent[..., latent_pointer:latent_pointer+2] / pt.sqrt(pt.sum(latent[..., latent_pointer:latent_pointer+2]**2, dim=2, keepdims=True) + 1e-16)
      latent_in.append(latent_sin_cos)
      print("Manipulate Sin Cos : ", latent_sin_cos.shape)
      remaining_size -= 2
      latent_pointer += 2

    if 'f' in self.latent_code and 'f_norm' not in self.latent_code and remaining_size > 0:
      #####################################
      ############### FORCE ###############
      #####################################
      latent_f = latent[..., latent_pointer:latent_pointer+3]
      latent_in.append(latent_f)
      print("Manipulate Force : ", latent_f.shape)
      remaining_size -= 1
      latent_pointer += 3

    if 'f' not in self.latent_code and 'f_norm' in self.latent_code and remaining_size > 0:
      #####################################
      ############# FORCE Norm ############
      #####################################
      latent_fnorm = latent[..., latent_pointer:latent_pointer+3] / pt.sqrt(pt.sum(latent[..., latent_pointer:latent_pointer+3]**2, dim=1, keepdims=True) + 1e-16)
      latent_in.append(latent_fnorm)
      print("Manipulate Force Norm : ", latent_fnorm.shape)
      remaining_size -= 3
      latent_pointer += 3

    if 'f' in self.latent_code and 'f_norm' in self.latent_code and remaining_size > 0:
      #####################################
      ######### FORCE & FORCE Norm ########
      #####################################
      latent_f = latent[..., latent_pointer:latent_pointer+3]
      latent_fnorm = latent_f / pt.sqrt(pt.sum(latent_f**2, dim=1, keepdims=True) + 1e-16)
      latent_in.append(latent_f)
      latent_in.append(latent_fnorm)
      print("Manipulate Force : ", latent_f.shape)
      print("Manipulate Force Norm : ", latent_fnorm.shape)
      remaining_size -= 3
      latent_pointer += 3

    latent_in = pt.cat(latent_in, dim=2)
    # print(latent_in)
    # print(latent_in.shape)
    return latent_in
