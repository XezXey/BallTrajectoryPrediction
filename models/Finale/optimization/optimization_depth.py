import torch as pt
import matplotlib.pyplot as plt
import numpy as np
import math

# TrajectoryOptimization Class
class TrajectoryOptimizationDepth(pt.nn.Module):
  def __init__(self, cam_params_dict, input_dict, latent_size, model_dict, pred_dict, latent_code):
    super(TrajectoryOptimizationDepth, self).__init__()
    self.cam_params_dict = cam_params_dict
    self.input_dict = input_dict
    self.model_dict = model_dict
    self.pred_dict = pred_dict
    self.latent_code = latent_code
    # Remove duplicate latent size
    if 'angle' in self.latent_code:
      # angle use only 1 dims then use sin(angle) and cos(angle)
      latent_size -= 1
    if 'f' in self.latent_code and 'fnorm' in self.latent_code:
      # force w/ fnorm use only 3 dims then use normalize for optimized f
      latent_size -= 3
    if 'sin_cos' in self.latent_code:
      # sin_cos use both then no duplicated
      latent_size -= 0
    self.latent_size = latent_size
    self.latent = pt.nn.ParameterList()

  def forward(self, in_f, lengths):
    latent = self.update_latent(lengths=lengths)
    latent = self.manipulate_latent(latent)
    in_f_ = pt.cat((in_f, latent), dim=2)
    pred_depth, (_, _) = self.model_dict['model_depth'](in_f=in_f_, lengths=lengths)
    return pred_depth

  def construct_latent(self, lengths):
    # This use EOT to split the latent. Stack and repeat to have the same shape with the "pred_xyz" 
    flag = self.pred_dict['model_flag']
    close = pt.isclose(flag, pt.tensor(1.), atol=5e-1)
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


  def update_latent(self, lengths):

    '''
    Return : latent with shape = (batchsize, seq_len, latent_size)
    '''
    flag = self.pred_dict['model_flag']
    lengths = lengths
    close = pt.isclose(flag, pt.tensor(1.), atol=5e-1)
    all_latent = pt.ones((flag.shape[0], flag.shape[1], self.latent_size)).cuda()
    for i in range(flag.shape[0]):
      # Each Trajectory
      where = pt.where(close[i] == True)[0]
      where = where[where < lengths[i]]
      if len(where) == 0:
        where = [0] + [flag.shape[1]]
      else:
        where = [0] + list(where.cpu().detach().numpy()+1) + [flag.shape[1]]
      for j in range(len(where)-1):
        all_latent[i][where[j]:where[j+1]] = self.latent[i][j].repeat(where[j+1] - where[j], 1)

      # all_latent.append(pt.cat(each_latent))
      # plt.plot(pt.cat(each_latent).cpu().detach().numpy(), 'r-o')
      # plt.plot(flag[i].cpu().detach().numpy(), 'g-o')
      # plt.plot(close[i].cpu().detach().numpy(), 'y-o')
      # plt.show()

    return all_latent

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
      #####################################
      ############# SIN & COS #############
      #####################################
      # latent_sin_cos = pt.cat((latent[..., [latent_pointer:latent_pointer+2]], latent[..., [latent_pointer:latent_pointer+2]])), dim=2)
      latent_sin_cos = latent[..., latent_pointer:latent_pointer+2] / pt.sqrt(pt.sum(latent[..., latent_pointer:latent_pointer+2]**2, dim=2, keepdims=True) + 1e-16)
      latent_in.append(latent_sin_cos)
      print("Manipulate Sin Cos : ", latent_sin_cos.shape)
      print("Manipulate Sin Cos : ", latent_sin_cos)
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
