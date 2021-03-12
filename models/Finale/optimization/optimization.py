
import torch as pt
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import os
from utils import utils_model as utils_model

# TrajectoryOptimization Class
class TrajectoryOptimization(pt.nn.Module):
  def __init__(self, cam_params_dict, gt_dict, latent_size, model_dict, n_refinement, pred_dict, latent_code, latent_transf):
    super(TrajectoryOptimization, self).__init__()
    self.cam_params_dict = cam_params_dict
    self.gt_dict = gt_dict
    self.model_dict = model_dict
    self.n_refinement = n_refinement
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
    self.latent_transf = latent_transf

  def construct_latent(self, lengths):
    '''
    Initialize the attribute name latent : Storing to be optimized latent with size of (#N flag prediction=1, latent_size)
    '''
    # This use EOT to split the latent. Stack and repeat to have the same shape with the "pred_xyz" 
    flag = self.pred_dict['model_flag']
    lengths = lengths
    if pt.max(lengths) >= pt.max(self.gt_dict['lengths']):
      # xyz features : Add first flag as zero to make these 2 seq have a same length
      flag = pt.cat((pt.zeros((flag.shape[0], 1, 1)).cuda(), flag), dim=1)
    close = pt.isclose(flag, pt.tensor(1.), atol=5e-1)
    for i in range(flag.shape[0]):
      where = pt.where(close[i] == True)[0]
      where = where[where < lengths[i]]
      if len(where) == 0:
        # Flag prediction goes wrong, 0 flag have been predicted ===> Create only 1 latent
        self.latent.append(pt.nn.Parameter(data=pt.randn(size=(1, self.latent_size), dtype=pt.float32).cuda(), requires_grad=True).cuda())
        # self.latent.append(pt.nn.Parameter(pt.zeros(1, self.latent_size, dtype=pt.float32).cuda(), requires_grad=True).cuda())
      else:
        # Flag prediction work properly ===> Create latents accroding to number of predicted flag=1
        n_latent = len(where)
        pt.manual_seed(np.random.randint(0, 999))
        self.latent.append(pt.nn.Parameter(data=pt.randn(size=(n_latent+1, self.latent_size), dtype=pt.float32).cuda(), requires_grad=True).cuda())

        # self.latent.append(pt.nn.Parameter(pt.zeros(n_latent+1, self.latent_size, dtype=pt.float32).cuda(), requires_grad=True).cuda())
    # for j in range(10):
      # print(pt.nn.Parameter(pt.randn(1, self.latent_size, dtype=pt.float32).cuda()))


  def update_latent(self, lengths):
    '''
    Update the latent by repeat in to have the same shape of input seq_len following the flag position
    Return : latent with shape = (batchsize, seq_len, latent_size)
    '''
    flag = self.pred_dict['model_flag']
    lengths = lengths
    if pt.max(lengths) >= pt.max(self.gt_dict['lengths']):
      # if input is xyz features : Add first flag as zero to make these 2 seq have a same length
      flag = pt.cat((pt.zeros((flag.shape[0], 1, 1)).cuda(), flag), dim=1)
    close = pt.isclose(flag, pt.tensor(1.), atol=5e-1)
    all_latent = []
    for i in range(flag.shape[0]):
      each_latent = []
      # Each Trajectory
      where = pt.where(close[i] == True)[0]
      where = where[where < lengths[i]]
      if len(where) == 0:
        where = [0] + [flag.shape[1]]
      else:
        where = [0] + list(where.cpu().detach().numpy()+1) + [flag.shape[1]]
      for j in range(len(where)-1):
        each_latent.append(self.latent[i][j].repeat(where[j+1] - where[j], 1))
      each_latent = pt.cat((each_latent))
      all_latent.append(each_latent)
    all_latent = pt.stack(all_latent, dim=0)

    return all_latent

  def manipulate_latent(self, latent):
    '''
    Manipulate the latent by transform it into the features that we've used since the model was trained,
      e.g. train with sin and cos ===> make the latent to be in sin and cos value
    Return : latent with shape = (batchsize, seq_len, latent_size) that in the features space that we've used while training the model
    '''
    remaining_size = self.latent_size
    latent_pointer = 0  # This is a column indexing striding along the features dimension to access each latent
    latent_in = []
    if 'angle' in self.latent_code and remaining_size > 0:
      #####################################
      ############### ANGLE ###############
      #####################################
      latent_angle = pt.cat((pt.sin(latent[..., [latent_pointer]] * math.pi/180.0), pt.cos(latent[..., [latent_pointer]] * math.pi/180.0)), dim=2)
      latent_in.append(latent_angle)
      print("Manipulate Angle : ", latent_angle.shape)
      remaining_size -= 1
      latent_pointer += 1

    if 'sin_cos' in self.latent_code and remaining_size > 0:
      #####################################
      ############# SIN & COS #############
      #####################################
      latent_sin_cos = latent[..., latent_pointer:latent_pointer+2] / pt.sqrt(pt.sum(latent[..., latent_pointer:latent_pointer+2]**2, dim=2, keepdims=True) + 1e-16)
      latent_in.append(latent_sin_cos)
      print("Manipulate Sin Cos : ", latent_sin_cos.shape)
      print("[#] Sanity check => Sin Cos : ", pt.all(pt.isclose(pt.sum(latent_sin_cos**2, dim=2, keepdims=True), pt.ones(size=latent_sin_cos.shape).cuda())).item())
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
      latent_fnorm = latent[..., latent_pointer:latent_pointer+3] / pt.sqrt(pt.sum(latent[..., latent_pointer:latent_pointer+3]**2, dim=2, keepdims=True) + 1e-16)
      latent_in.append(latent_fnorm)
      print("Manipulate Force Norm : ", latent_fnorm.shape)
      remaining_size -= 3
      latent_pointer += 3

    if 'f' in self.latent_code and 'f_norm' in self.latent_code and remaining_size > 0:
      #####################################
      ######### FORCE & FORCE Norm ########
      #####################################
      latent_f = latent[..., latent_pointer:latent_pointer+3]
      latent_fnorm = latent_f / pt.sqrt(pt.sum(latent_f**2, dim=2, keepdims=True) + 1e-16)
      latent_in.append(latent_f)
      latent_in.append(latent_fnorm)
      print("Manipulate Force : ", latent_f.shape)
      print("Manipulate Force Norm : ", latent_fnorm.shape)
      remaining_size -= 3
      latent_pointer += 3

    latent_in = pt.cat(latent_in, dim=2)
    return latent_in

class TrajectoryOptimizationRefinement(TrajectoryOptimization):

  def forward(self, in_f, lengths, model_encoder=None):
    # Expand latent into specific lengths and shape
    latent = self.update_latent(lengths)
    # Transform a latent into a correct features space
    latent = self.manipulate_latent(latent)
    # Concant with in_f
    in_f_ = pt.cat((in_f, latent), dim=2)
    # If latent_transf operation is used
    if self.latent_transf is not None:
      in_f_tranf = utils_model.latent_transform(in_f = in_f_)
    else: in_f_tranf = in_f_

    if model_encoder is not None:
      in_f_final = model_encoder(in_f_tranf)
    else:
      in_f_final = in_f_tranf

    # Make a prediction
    for idx in range(self.n_refinement):
      pred_refinement, (_, _) = self.model_dict['model_refinement_{}'.format(idx)](in_f=in_f_final, lengths=lengths)
    return pred_refinement


# TrajectoryOptimization Class
class TrajectoryOptimizationDepth(pt.nn.Module):

  def forward(self, in_f, lengths):
    # Expand latent into specific lengths and shape
    latent = self.update_latent(lengths=lengths)
    # Transform a latent into a correct features space
    latent = self.manipulate_latent(latent)
    # Concant with in_f
    in_f_ = pt.cat((in_f, latent), dim=2)
    # Make a prediction
    pred_depth, (_, _) = self.model_dict['model_depth'](in_f=in_f_, lengths=lengths)
    return pred_depth

class Analyzer(TrajectoryOptimization):

  def forward(self, in_f, lengths):
    # If latent_transf operation is used
    if self.latent_transf is not None:
      in_f_ = utils_model.latent_transform(in_f = in_f)
    else: in_f_ = in_f

    # Make a prediction
    for idx in range(self.n_refinement):
      pred_refinement, (_, _) = self.model_dict['model_refinement_{}'.format(idx)](in_f=in_f_, lengths=lengths)
    return pred_refinement
