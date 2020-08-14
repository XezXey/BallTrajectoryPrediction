from __future__ import print_function
# Import libs
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import glob
import os
import argparse
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import PIL.Image
import io
import wandb
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from mpl_toolkits import mplot3d
import json
# Animated visualization
from utils.animated_visualization import trajectory_animation
# Dataloader
from utils.dataloader import TrajectoryDataset
# Models
from models.rnn_model import RNN
from models.lstm_model import LSTM
from models.bilstm_model import BiLSTM
from models.bigru_model import BiGRU
from models.gru_model import GRU
from models.bigru_model_residual_list import BiGRUResidualList
from models.bigru_model_residual_add import BiGRUResidualAdd

def make_visualize(output_test_xyz, output_trajectory_test_xyz, output_trajectory_test_startpos, input_trajectory_test_temp, input_trajectory_test_lengths, output_trajectory_test_mask, visualization_path, mae_loss_trajectory, mae_loss_3axis, trajectory_type, animation_visualize_flag, input_eot):
    # Visualize by make a subplots of trajectory
    n_vis = 5
    fig = make_subplots(rows=n_vis*2, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter3d'}], [{'colspan':2}, None]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
    # Random the index the be visualize
    vis_idx = np.random.randint(low=0, high=input_trajectory_test_startpos.shape[0], size=(n_vis))
    # Visualize a trajectory
    visualize_trajectory(input=input_trajectory_test_temp, output=pt.mul(output_test_xyz, output_trajectory_test_mask[..., :-1]), trajectory_gt=output_trajectory_test_xyz[..., :-1], trajectory_startpos=output_trajectory_test_startpos[..., :-1], lengths=input_trajectory_test_lengths, mask=output_trajectory_test_mask[..., :-1], fig=fig, flag='Test', n_vis=n_vis, mae_loss_trajectory=mae_loss_trajectory.cpu().detach().numpy(), mae_loss_3axis=mae_loss_3axis.cpu().detach().numpy(), vis_idx=vis_idx, input_eot=input_eot)
    # Adjust the layout/axis
    # AUTO SCALED/PITCH SCALED
    fig.update_layout(height=2048, width=1500, autosize=True, title="Testing on {} trajectory: Trajectory Visualization with EOT flag(Col1=PITCH SCALED, Col2=AUTO SCALED)".format(trajectory_type))
    fig = visualize_layout_update(fig=fig, n_vis=n_vis)
    fig.show()
    if animation_visualize_flag:
      trajectory_animation(output_xyz=pt.mul(output_test_xyz, output_trajectory_test_mask[..., :-1]), gt_xyz=output_trajectory_test_xyz[..., :-1], input_uv=input_trajectory_test_temp, lengths=input_trajectory_test_lengths, mask=output_trajectory_test_mask[..., :-1], n_vis=n_vis, html_savepath=visualization_path, vis_idx=vis_idx)
    input("Continue plotting...")

def visualize_layout_update(fig=None, n_vis=7):
  # Save to html file and use wandb to log the html and display (Plotly3D is not working)
  # fig.update_layout(height=1920, width=1080, margin=dict(l=0, r=0, b=5,t=5,pad=1), autosize=False)
  for i in range(n_vis*2):
    if i%2==0:
      # Set the figure in column 1 (fig0, 2, 4, ...) into a pitch scaled
      fig['layout']['scene{}'.format(i+1)].update(xaxis=dict(nticks=10, range=[-5, 5],), yaxis = dict(nticks=5, range=[-2, 4],), zaxis = dict(nticks=10, range=[-5, 5],),)
    fig['layout']['scene{}'.format(i+1)]['camera'].update(projection=dict(type="perspective"))
  return fig

def visualize_trajectory(input, output, trajectory_gt, trajectory_startpos, lengths, mask, mae_loss_trajectory, mae_loss_3axis, vis_idx, input_eot, fig=None, flag='test', n_vis=5):
  # marker_dict for contain the marker properties
  marker_dict_gt = dict(color='rgba(0, 0, 255, 0.2)', size=3)
  marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=3)
  # Visualize the displacement
  marker_dict_u = dict(color='rgba(255, 0, 0, 0.7)', size=4)
  marker_dict_v = dict(color='rgba(0, 255, 0, 0.7)', size=4)
  marker_dict_depth = dict(color='rgba(0, 0, 255, 0.5)', size=4)
  marker_dict_eot = dict(color='rgba(0, 0, 255, 0.5)', size=3)

  # MAE Loss
  # detach() for visualization
  input = input.cpu().detach().numpy()
  input_eot = input_eot.cpu().detach().numpy()
  input_eot *= 0.1
  output = output.cpu().detach().numpy()
  trajectory_gt = trajectory_gt.cpu().detach().numpy()
  # Iterate to plot each trajectory
  count = 1
  for idx, i in enumerate(vis_idx):
    for col_idx in range(1, 3):
      fig.add_trace(go.Scatter3d(x=output[i][:lengths[i]+1, 0], y=output[i][:lengths[i]+1, 1], z=output[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_pred, name="{}-Estimated Trajectory [{}], MSE = {:.3f}, MAE_trajectory = {:.3f}, MAE_3axis = {}".format(flag, i, MSELoss(pt.tensor(output[i]).to(device), pt.tensor(trajectory_gt[i]).to(device), mask=mask[i]), mae_loss_trajectory[i], mae_loss_3axis[i, :])), row=idx+count, col=col_idx)
      fig.add_trace(go.Scatter3d(x=trajectory_gt[i][:lengths[i]+1, 0], y=trajectory_gt[i][:lengths[i]+1, 1], z=trajectory_gt[i][:lengths[i]+1, 2], mode='markers', marker=marker_dict_gt, name="{}-Ground Truth Trajectory [{}]".format(flag, i)), row=idx+count, col=col_idx)
    count +=1

  # Iterate to plot each displacement of (u, v, depth)
  for idx, i in enumerate(vis_idx):
    col_idx = 1
    row_idx = (idx*2) + 2
    fig.add_trace(go.Scatter(x=np.arange(input_eot[i][:lengths[i]].shape[0]), y=input_eot[i][:lengths[i]], marker=marker_dict_eot, mode='markers+lines', name='EOT'), row=row_idx, col=col_idx)
    fig.add_trace(go.Scatter(x=np.arange(input[i][:lengths[i]+1, 0].shape[0]), y=np.diff(input[i][:lengths[i]+1, 0]), marker=marker_dict_u, mode='lines', name='U'), row=row_idx, col=col_idx)
    fig.add_trace(go.Scatter(x=np.arange(input[i][:lengths[i]+1, 1].shape[0]), y=np.diff(input[i][:lengths[i]+1, 1]), marker=marker_dict_v, mode='lines', name='V'), row=row_idx, col=col_idx)
    # fig.add_trace(go.Scatter(x=np.arange(input[i][:lengths[i], 2].shape[0]), y=input[i][:lengths[i], 2], marker=marker_dict_depth, mode='lines'), row=row_idx, col=col_idx)

  plotly.offline.plot(fig, filename='./{}/trajectory_visualization_depth.html'.format(args.visualization_path), auto_open=True)
  return vis_idx

def compute_gravity_constraint_penalize(output, trajectory_gt, mask, lengths):
  gravity_constraint_penalize = 0
  # Gaussian blur kernel for not get rid of the input information
  gaussian_blur = pt.tensor([0.25, 0.5, 0.25], dtype=pt.float32).view(1, 1, -1).to(device)
  # Kernel weight for performing a finite difference
  kernel_weight = pt.tensor([-1., 0., 1.], dtype=pt.float32).view(1, 1, -1).to(device)
  # Apply Gaussian blur and finite difference to trajectory_gt
  for i in range(trajectory_gt.shape[0]):
    if trajectory_gt[i][:lengths[i]+1, 1].shape[0] < 6:
      print("The trajectory is too shorter to perform a convolution")
      continue
    trajectory_gt_yaxis_1st_gaussian_blur = pt.nn.functional.conv1d(trajectory_gt[i][:lengths[i]+1, 1].view(1, 1, -1), gaussian_blur)
    trajectory_gt_yaxis_1st_finite_difference = pt.nn.functional.conv1d(trajectory_gt_yaxis_1st_gaussian_blur, kernel_weight)
    trajectory_gt_yaxis_2nd_gaussian_blur = pt.nn.functional.conv1d(trajectory_gt_yaxis_1st_finite_difference, gaussian_blur)
    trajectory_gt_yaxis_2nd_finite_difference = pt.nn.functional.conv1d(trajectory_gt_yaxis_2nd_gaussian_blur, kernel_weight)
    # Apply Gaussian blur and finite difference to trajectory_gt
    output_yaxis_1st_gaussian_blur = pt.nn.functional.conv1d(output[i][:lengths[i]+1, 1].view(1, 1, -1), gaussian_blur)
    output_yaxis_1st_finite_difference = pt.nn.functional.conv1d(output_yaxis_1st_gaussian_blur, kernel_weight)
    output_yaxis_2nd_gaussian_blur = pt.nn.functional.conv1d(output_yaxis_1st_finite_difference, gaussian_blur)
    output_yaxis_2nd_finite_difference = pt.nn.functional.conv1d(output_yaxis_2nd_gaussian_blur, kernel_weight)
    # Compute the penalize term
    # print(trajectory_gt_yaxis_2nd_finite_difference, output_yaxis_2nd_finite_difference)
    gravity_constraint_penalize += ((pt.sum(trajectory_gt_yaxis_2nd_finite_difference - output_yaxis_2nd_finite_difference)))**2
  return gravity_constraint_penalize

def MSELoss(output, trajectory_gt, mask, lengths=None, delmask=True):
  # Adding on ground penalize
  if lengths is None :
    gravity_constraint_penalize = pt.tensor(0).to(device)
  else:
    gravity_constraint_penalize = compute_gravity_constraint_penalize(output=output.clone(), trajectory_gt=trajectory_gt.clone(), mask=mask, lengths=lengths)
  mse_loss = (pt.sum((((trajectory_gt - output))**2) * mask) / pt.sum(mask)) # + gravity_constraint_penalize
  return mse_loss

def evaluateModel(output, trajectory_gt, mask, lengths, threshold=1, delmask=True):
  mae_loss_3axis = pt.sum(((pt.abs(trajectory_gt - output)) * mask), axis=1) / pt.sum(mask, axis=1)
  mae_loss_trajectory = pt.sum(mae_loss_3axis, axis=1) / 3
  accepted_3axis_loss = pt.sum((pt.sum(mae_loss_3axis < threshold, axis=1) == 3))
  print("Accepted 3-Axis(X, Y, Z) loss < {} : {}".format(threshold, accepted_3axis_loss))
  accepted_trajectory_loss = pt.sum(mae_loss_trajectory < threshold)
  print("Accepted trajectory loss < {} : {}".format(threshold, accepted_trajectory_loss))
  return accepted_3axis_loss, accepted_trajectory_loss, mae_loss_trajectory, mae_loss_3axis

def projectToWorldSpace(screen_space, depth, projection_matrix, camera_to_world_matrix, width, height):
  depth = depth.view(-1)
  screen_width = width
  screen_height = height
  # Screnn space -> NDC space
  # print("SCREEN : ", screen_space)
  # print("DEPTH : ", depth)
  screen_space = pt.div(screen_space, pt.tensor([screen_width, screen_height]).to(device)) # Normalize : (width, height) -> (-1, 1)
  # print("NDC : ", screen_space)
  # NDC space -> CAMERA space
  screen_space = (screen_space * 2.0) - pt.ones(size=(screen_space.size()), dtype=pt.float32).to(device) # Normalize : (width, height) -> (-1, 1)
  # print("SCREEN : ", screen_space)
  screen_space = (screen_space.t() * depth).t()   # Normalize : (-1, 1) -> (-depth, depth)
  # print("CAMERA : ", screen_space)
  screen_space = pt.stack((screen_space[:, 0], screen_space[:, 1], depth, pt.ones(depth.shape[0], dtype=pt.float32).to(device)), axis=1) # Stack the screen with depth and w ===> (x, y, depth, 1)
  # print("CAMERA : ", screen_space)
  screen_space = (camera_to_world_matrix @ projection_matrix @ screen_space.t()).t() # Reprojected
  # print("WORLD : ", screen_space)
  return screen_space[:, :3]

def cumsum_trajectory(output, trajectory, trajectory_startpos):
  '''
  Perform a cummulative summation to the output
  Argument :
  1. output : The displacement from the network with shape = (batch_size, sequence_length,)
  2. trajectory : The input_trajectory (displacement of u, v) with shape = (batch_size, sequence_length, 2)
  3. trajectory_startpos : The start position of input trajectory with shape = (batch_size, 1, )
  Output :
  1. output : concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1,)
  2. trajectory_temp : u, v by concat with startpos and perform cumsum with shape = (batch_size, sequence_length+1, 2)
  '''
  # Apply cummulative summation to output
  # trajectory_temp : concat with startpos and stack back to (batch_size, sequence_length+1, 2)
  trajectory_temp = pt.stack([pt.cat([trajectory_startpos[i][:, :2], trajectory[i].clone().detach()]) for i in range(trajectory_startpos.shape[0])])
  # trajectory_temp : perform cumsum along the sequence_length axis
  trajectory_temp = pt.cumsum(trajectory_temp, dim=1)
  # output : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output = pt.stack([pt.cat([trajectory_startpos[i][:, -1].view(-1, 1), output[i]]) for i in range(trajectory_startpos.shape[0])])
  # output : perform cumsum along the sequence_length axis
  output = pt.cumsum(output, dim=1)
  # print(output.shape, trajectory_temp.shape)
  return output, trajectory_temp

def predict(output_trajectory_test, output_trajectory_test_mask, output_trajectory_test_lengths, output_trajectory_test_startpos, output_trajectory_test_xyz, input_trajectory_test, input_trajectory_test_mask, input_trajectory_test_lengths, input_trajectory_test_startpos, model, hidden, cell_state, projection_matrix, camera_to_world_matrix, trajectory_type, threshold, width, height, animation_visualize_flag=False, visualize_trajectory_flag=True, visualization_path='./visualize_html/'):
  # Testing RNN/LSTM model
  # Initial hidden layer for the first RNN Cell
  model.eval()
  # Test a model on a testing batch
  # Forward PASSING
  # Forward pass for testing a model
  output_test, (_, _) = model(input_trajectory_test, hidden, cell_state, lengths=input_trajectory_test_lengths)
  # (This step we get the displacement of depth by input the displacement of u and v)
  # Apply cummulative summation to output using cumsum_trajectory function
  output_test, input_trajectory_test_temp = cumsum_trajectory(output=output_test, trajectory=input_trajectory_test[..., :-1], trajectory_startpos=input_trajectory_test_startpos[..., :-1])
  # Project the (u, v, depth) to world space
  output_test_xyz = pt.stack([projectToWorldSpace(screen_space=input_trajectory_test_temp[i], depth=output_test[i], projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, width=width, height=height) for i in range(output_test.shape[0])])
  # Calculate loss of unprojected trajectory
  test_loss = MSELoss(output=output_test_xyz, trajectory_gt=output_trajectory_test_xyz[..., :-1], mask=output_trajectory_test_mask[..., :-1], lengths=output_trajectory_test_lengths)
  # Calculate loss per trajectory
  accepted_3axis_loss, accepted_trajectory_loss, mae_loss_trajectory, mae_loss_3axis = evaluateModel(output=output_test_xyz, trajectory_gt=output_trajectory_test_xyz[..., :-1], mask=output_trajectory_test_mask[..., :-1], lengths=output_trajectory_test_lengths, threshold=threshold)

  print('===>Test Loss : {:.3f}'.format(test_loss.item()))
  if visualize_trajectory_flag == True:
      make_visualize(output_test_xyz=output_test_xyz, output_trajectory_test_xyz=output_trajectory_test_xyz, output_trajectory_test_startpos=output_trajectory_test_startpos, input_trajectory_test_lengths=input_trajectory_test_lengths, input_trajectory_test_temp=input_trajectory_test_temp, output_trajectory_test_mask=output_trajectory_test_mask, visualization_path=visualization_path, mae_loss_trajectory=mae_loss_trajectory, mae_loss_3axis=mae_loss_3axis, trajectory_type=trajectory_type, animation_visualize_flag=animation_visualize_flag, input_eot=input_trajectory_test[..., -1])

  return accepted_3axis_loss, accepted_trajectory_loss

def initialize_folder(path):
  if not os.path.exists(path):
      os.makedirs(path)

def collate_fn_padd(batch):
    '''
    Padding batch of variable length
    '''
    padding_value = -10
    ## Get sequence lengths
    lengths = pt.tensor([trajectory[1:, :].shape[0] for trajectory in batch])
    # Input features : columns 4-5 contain u, v in screen space
    ## Padding 
    input_batch = [pt.Tensor(trajectory[1:, [3, 4, -1]]) for trajectory in batch] # Mocap
    # input_batch = [pt.Tensor(trajectory[1:, [4, 5, -2]]) for trajectory in batch] # Unity (4, 5, -2) = (u, v, end_of_trajectory)
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=padding_value)
    ## Retrieve initial position (u, v, depth)
    input_startpos = pt.stack([pt.Tensor(trajectory[0, [3, 4, 5, -1]]) for trajectory in batch]) # Mocap
    # input_startpos = pt.stack([pt.Tensor(trajectory[0, [4, 5, 6, -2]]) for trajectory in batch]) # Unity
    input_startpos = pt.unsqueeze(input_startpos, dim=1)
    ## Compute mask
    input_mask = (input_batch != padding_value)

    # Output features : columns 6 cotain depth from camera to projected screen
    ## Padding
    output_batch = [pt.Tensor(trajectory[:, [5, -1]]) for trajectory in batch] # Mocap
    # output_batch = [pt.Tensor(trajectory[:, [6, -2]]) for trajectory in batch] # Unity
    output_batch = pad_sequence(output_batch, batch_first=True)
    ## Retrieve initial position
    output_startpos = pt.stack([pt.Tensor(trajectory[0, [0, 1, 2, -1]]) for trajectory in batch]) # Mocap
    # output_startpos = pt.stack([pt.Tensor(trajectory[0, [0, 1, 2, -2]]) for trajectory in batch]) # Unity
    output_startpos = pt.unsqueeze(output_startpos, dim=1)
    ## Retrieve the x, y, z in world space for compute the reprojection error (x', y', z' <===> x, y, z)
    output_xyz = [pt.Tensor(trajectory[:, [0, 1, 2, -1]]) for trajectory in batch] # Mocap
    # output_xyz = [pt.Tensor(trajectory[:, [0, 1, 2, -2]]) for trajectory in batch] # Unity
    output_xyz = pad_sequence(output_xyz, batch_first=True, padding_value=padding_value)
    ## Compute mask
    output_mask = (output_xyz != padding_value)
    ## Compute cummulative summation to form a trajectory from displacement every columns except the end_of_trajectory
    # print(output_xyz[..., :-1].shape, pt.unsqueeze(output_xyz[..., -1], dim=2).shape)
    output_xyz = pt.cat((pt.cumsum(output_xyz[..., :-1], dim=1), pt.unsqueeze(output_xyz[..., -1], dim=2)), dim=2)

    # print("Mask shape : ", mask.shape)
    return {'input':[input_batch, lengths, input_mask, input_startpos],
            'output':[output_batch, lengths, output_mask, output_startpos, output_xyz]}

def get_model(input_size, output_size, model_arch):
  if model_arch=='gru':
    rnn_model = GRU(input_size=input_size, output_size=output_size)
  elif model_arch=='bigru':
    rnn_model = BiGRU(input_size=input_size, output_size=output_size)
  elif model_arch=='bigru_residual_list':
    rnn_model = BiGRUResidualList(input_size=input_size, output_size=output_size)
  elif model_arch=='bigru_residual_add':
    rnn_model = BiGRUResidualAdd(input_size=input_size, output_size=output_size)
  elif model_arch=='lstm':
    rnn_model = LSTM(input_size=input_size, output_size=output_size)
  elif model_arch=='bigru':
    rnn_model = BiLSTM(input_size=input_size, output_size=output_size)

  return rnn_model

if __name__ == '__main__':
  print('[#]Training : Trajectory Estimation')
  # Argumentparser for input
  parser = argparse.ArgumentParser(description='Predict the 3D Trajectory')
  parser.add_argument('--dataset_test_path', dest='dataset_test_path', type=str, help='Path to testing set', required=True)
  parser.add_argument('--batch_size', dest='batch_size', type=int, help='Size of batch', default=50)
  parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, help='Type of trajectory(Rolling, Projectile, MagnusProjectile)', default='Projectile')
  parser.add_argument('--visualize', dest='visualize_trajectory_flag', help='Visualize the trajectory', action='store_true')
  parser.add_argument('--no_visualize', dest='visualize_trajectory_flag', help='Visualize the trajectory', action='store_false')
  parser.add_argument('--pretrained_model_path', dest='pretrained_model_path', type=str, help='Path to load a trained model checkpoint', default=None)
  parser.add_argument('--visualization_path', dest='visualization_path', type=str, help='Path to visualization directory', default='./visualize_html/')
  parser.add_argument('--cam_params_file', dest='cam_params_file', type=str, help='Path to camera parameters file(Intrinsic/Extrinsic)')
  parser.add_argument('--cuda_device_num', dest='cuda_device_num', type=int, help='Provide cuda device number', default=0)
  parser.add_argument('--threshold', dest='threshold', type=float, help='Provide the error threshold of reconstructed trajectory', default=0.8)
  parser.add_argument('--no_animation', dest='animation_visualize_flag', help='Animated visualize flag', action='store_false')
  parser.add_argument('--animation', dest='animation_visualize_flag', help='Animated visualize flag', action='store_true')
  parser.add_argument('--model_arch', dest='model_arch', type=str, help='Input model architecture (gru, bigru, lstm, bilstm)', required=True)
  args = parser.parse_args()
  # Initialize folder
  initialize_folder(args.visualization_path)

  # GPU initialization
  if pt.cuda.is_available():
    pt.cuda.set_device(args.cuda_device_num)
    device = pt.device('cuda')
    print('[%]GPU Enabled')
  else:
    device = pt.device('cpu')
    print('[%]GPU Disabled, CPU Enabled')

  # Load camera parameters : ProjectionMatrix and WorldToCameraMatrix
  with open(args.cam_params_file) as cam_params_json:
    cam_params_file = json.load(cam_params_json)
    cam_params = dict({'projectionMatrix':cam_params_file['mainCameraParams']['projectionMatrix'], 'worldToCameraMatrix':cam_params_file['mainCameraParams']['worldToCameraMatrix'], 'width':cam_params_file['mainCameraParams']['width'], 'height':cam_params_file['mainCameraParams']['height']})
  projection_matrix = np.array(cam_params['projectionMatrix']).reshape(4, 4)
  projection_matrix = pt.tensor([projection_matrix[0, :], projection_matrix[1, :], projection_matrix[3, :], [0, 0, 0, 1]], dtype=pt.float32)
  projection_matrix = pt.inverse(projection_matrix).to(device)
  camera_to_world_matrix = pt.inverse(pt.tensor(cam_params['worldToCameraMatrix']).view(4, 4)).to(device)
  width = cam_params['width']
  height = cam_params['height']

  # Create Datasetloader for test
  trajectory_test_dataset = TrajectoryDataset(dataset_path=args.dataset_test_path, trajectory_type=args.trajectory_type)
  trajectory_test_dataloader = DataLoader(trajectory_test_dataset, batch_size=args.batch_size, num_workers=10, shuffle=False, collate_fn=collate_fn_padd, pin_memory=True)#, drop_last=True)

  # Dataset format
  # Trajectory path : (x0, y0) ... (xn, yn)
  print("======================================================Summary Batch (batch_size = {})=========================================================================".format(args.batch_size))
  for key, batch in enumerate(trajectory_test_dataloader):
    print("Input batch [{}] : batch={}, lengths={}, mask={}, initial position={}".format(key, batch['input'][0].shape, batch['input'][1].shape, batch['input'][2].shape, batch['input'][3].shape))
    print("Output batch [{}] : batch={}, lengths={}, mask={}, initial position={}".format(key, batch['output'][0].shape, batch['output'][1].shape, batch['output'][2].shape, batch['output'][3].shape))
    # Test RNN/LSTM Step
    # 1.Pack the padded
    packed = pt.nn.utils.rnn.pack_padded_sequence(batch['input'][0], batch_first=True, lengths=batch['input'][1], enforce_sorted=False)
    # 2.RNN/LSTM model
    # 3.Unpack the packed
    unpacked = pt.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, padding_value=-1)
    print("Unpacked equality : ", pt.eq(batch['input'][0], unpacked[0]).all())
    print("===============================================================================================================================================================")

  # Model definition
  n_output = 1 # Contain the depth information of the trajectory
  n_input = 3 # Contain following this trajectory parameters (u, v) position from tracking
  print('[#]Model Architecture')
  rnn_model = get_model(input_size=n_input, output_size=n_output, model_arch=args.model_arch)
  if args.pretrained_model_path is None:
    print('===>No pre-trained model to load')
    print('EXIT...')
    exit()
  else:
    print('===>Load trained model')
    rnn_model.load_state_dict(pt.load(args.pretrained_model_path, map_location=device))
  rnn_model = rnn_model.to(device)
  print(rnn_model)

  hidden = rnn_model.initHidden(batch_size=args.batch_size)
  cell_state = rnn_model.initCellState(batch_size=args.batch_size)
  # Test a model iterate over dataloader to get each batch and pass to predict function
  n_accepted_3axis_loss = 0
  n_accepted_trajectory_loss = 0
  n_trajectory = 0
  for batch_idx, batch_test in enumerate(trajectory_test_dataloader):
    print("[#]Batch-{}".format(batch_idx))
    # Testing set (Each index in batch_test came from the collate_fn_padd)
    input_trajectory_test = batch_test['input'][0].to(device)
    input_trajectory_test_lengths = batch_test['input'][1].to(device)
    input_trajectory_test_mask = batch_test['input'][2].to(device)
    input_trajectory_test_startpos = batch_test['input'][3].to(device)
    output_trajectory_test = batch_test['output'][0].to(device)
    output_trajectory_test_lengths = batch_test['output'][1].to(device)
    output_trajectory_test_mask = batch_test['output'][2].to(device)
    output_trajectory_test_startpos = batch_test['output'][3].to(device)
    output_trajectory_test_xyz = batch_test['output'][4].to(device)

    # Call function to test
    accepted_3axis_loss, accepted_trajectory_loss = predict(output_trajectory_test=output_trajectory_test, output_trajectory_test_mask=output_trajectory_test_mask,
                                             output_trajectory_test_lengths=output_trajectory_test_lengths, output_trajectory_test_startpos=output_trajectory_test_startpos, output_trajectory_test_xyz=output_trajectory_test_xyz,
                                             input_trajectory_test=input_trajectory_test, input_trajectory_test_mask = input_trajectory_test_mask,
                                             input_trajectory_test_lengths=input_trajectory_test_lengths, input_trajectory_test_startpos=input_trajectory_test_startpos,
                                             model=rnn_model, hidden=hidden, cell_state=cell_state, visualize_trajectory_flag=args.visualize_trajectory_flag,
                                             projection_matrix=projection_matrix, camera_to_world_matrix=camera_to_world_matrix, trajectory_type=args.trajectory_type, threshold=args.threshold, animation_visualize_flag=args.animation_visualize_flag, width=width, height=height)
    n_accepted_3axis_loss += accepted_3axis_loss
    n_accepted_trajectory_loss += accepted_trajectory_loss
    n_trajectory += input_trajectory_test.shape[0]


  print("="*100)
  print("[#]Summary")
  print("Accepted trajectory by MAE Loss : {} from {}".format(n_accepted_trajectory_loss, n_trajectory))
  print("Accepted trajectory by 3axis MAE Loss : {} from {}".format(n_accepted_3axis_loss, n_trajectory))
  print("="*100)

  print("[#] Done")
