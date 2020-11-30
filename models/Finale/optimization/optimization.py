import torch as pt
import matplotlib.pyplot as plt
import numpy as np

# TrajectoryOptimization Class
class TrajectoryOptimization(pt.nn.Module):
  def __init__(self, pred_xyz, cam_params_dict, gt_dict, latent_size, model_dict, n_refinement, pred_dict):
    super(TrajectoryOptimization, self).__init__()
    self.pred_xyz = pred_xyz
    self.cam_params_dict = cam_params_dict
    self.gt_dict = gt_dict
    self.model_dict = model_dict
    self.n_refinement = n_refinement
    self.pred_dict = pred_dict
    # self.latent = pt.nn.Parameter(pt.rand(pred_xyz.shape[0], 1, latent_size, dtype=pt.float32).cuda(), requires_grad=True).cuda()
    # self.latent = pt.nn.ParameterList(self.construct_latent())
    self.latent = pt.nn.ParameterList()
    self.construct_latent()

  def forward(self, xyz):
    # Repeat
    # latent = self.latent.repeat(1, self.pred_xyz.shape[1], 1).cuda()
    # Latent by EOT-seperation
    latent = self.update_latent()
    in_f = pt.cat((xyz, pt.sin(latent), pt.cos(latent)), dim=2)
    for idx in range(self.n_refinement):
      pred_refinement, (_, _) = self.model_dict['model_refinement_{}'.format(idx)](in_f=in_f, lengths=self.gt_dict['lengths'])
      xyz = xyz + pred_refinement
      in_f = pt.cat((xyz, pt.sin(latent), pt.cos(latent)), dim=2)
    return xyz

  def construct_latent(self):
    # This use EOT to split the latent. Stack and repeat to have the same shape with the "pred_xyz" 
    flag = self.pred_dict['model_flag']
    flag = pt.cat((pt.zeros((flag.shape[0], 1, 1)).cuda(), flag), dim=1)
    close = pt.isclose(flag, pt.tensor(1.), atol=2e-1)
    latent_list = []
    for i in range(flag.shape[0]):
      where = pt.where(close[i] == True)[0]
      if len(where) == 0:
        # Flag prediction goes wrong
        latent = pt.nn.Parameter(pt.rand(1, 1, dtype=pt.float32).cuda(), requires_grad=True).cuda()
        self.latent.append(pt.nn.Parameter(pt.rand(1, 1, dtype=pt.float32).cuda(), requires_grad=True).cuda())
      else:
        # Flag prediction work properly
        n_latent = len(where)
        latent = pt.nn.Parameter(pt.rand(n_latent+1, 1, dtype=pt.float32).cuda(), requires_grad=True).cuda()
        self.latent.append(pt.nn.Parameter(pt.rand(n_latent+1, 1, dtype=pt.float32).cuda(), requires_grad=True).cuda())

      # print(latent.shape)
      # latent_list.append(latent)
    # latent = pt.stack(latent_list)
    # print(latent.shape)
    # self.register_parameter('latent', latent)
    return latent_list

  def update_latent(self):
    '''
    Return : latent with shape = (batchsize, seq_len, latent_size)
    '''
    flag = self.pred_dict['model_flag']
    flag = pt.cat((pt.zeros((flag.shape[0], 1, 1)).cuda(), flag), dim=1)
    close = pt.isclose(flag, pt.tensor(1.), atol=2e-1)
    # all_latent = []
    all_latent = pt.ones(flag.shape).cuda()
    for i in range(flag.shape[0]):
      # Each Trajectory
      if len(pt.where(close[i] == True)[0]) == 0:
        where = [0] + [flag.shape[1]]
      else:
        where = [0] + list(pt.where(close[i] == True)[0].cpu().detach().numpy()+1) + [flag.shape[1]]
      # each_latent = []
      for j in range(len(where)-1):
        all_latent[i][where[j]:where[j+1]] = self.latent[i][j].repeat(where[j+1] - where[j], 1)
        # Each EOT-seperation
        # print(self.latent[j])
        # print(self.latent[i][j])
        # each_latent.append(self.latent[i][j].repeat(where[j+1] - where[j], 1))

      # all_latent.append(pt.cat(each_latent))
      # plt.plot(pt.cat(each_latent).cpu().detach().numpy(), 'r-o')
      # plt.plot(flag[i].cpu().detach().numpy(), 'g-o')
      # plt.plot(close[i].cpu().detach().numpy(), 'y-o')
      # plt.show()

    # all_latent = pt.stack(all_latent)
    # print(all_latent.shape)
    # exit()
    # return all_latent
    return all_latent
