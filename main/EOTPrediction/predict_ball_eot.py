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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
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

def make_visualize(input_trajectory_test, output_test_eot, output_trajectory_test_startpos, input_trajectory_test_lengths, output_trajectory_test_maks, visualization_path):
  # Visualize by make a subplots of trajectory 
  n_vis = 7
  if n_vis > args.batch_size:
    n_vis = args.batch_size
  if n_vis > input_trajectory_test.shape[0]:
    n_vis = input_trajectory_test.shape[0]
  # Random the index the be visualize
  test_vis_idx = np.random.choice(a=np.arange(input_trajectory_test.shape[0]), size=(n_vis), replace=False)
  # Visualize the End of trajectory(EOT) flag
  fig_eot = make_subplots(rows=n_vis, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter'}]]*n_vis, horizontal_spacing=0.05, vertical_spacing=0.01)
  visualize_eot(input_uv=input_trajectory_test, output_eot=output_test_eot.clone(), eot_gt=output_trajectory_test_uv[..., -1], eot_startpos=output_trajectory_test_startpos[..., -1], lengths=input_trajectory_test_lengths, mask=output_trajectory_test_mask[..., -1], fig=fig_eot, flag='test', n_vis=n_vis, vis_idx=test_vis_idx)
  plotly.offline.plot(fig_eot, filename='./{}/EndOfTrajectory_flag_visualization.html'.format(visualization_path), auto_open=True)

def visualize_eot(input_uv, output_eot, eot_gt, eot_startpos, lengths, mask, vis_idx, fig=None, flag='train', n_vis=5):
  # Add the feature dimension using unsqueeze
  eot_gt = pt.unsqueeze(eot_gt, dim=2)
  mask = pt.unsqueeze(mask, dim=2)
  eot_startpos = pt.unsqueeze(eot_startpos, dim=2)
  # eot_gt : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output_eot = pt.stack([pt.cat([eot_startpos[i], output_eot[i]]) for i in range(eot_startpos.shape[0])])
  output_eot *= mask
  eot_gt *= mask
  # print(pt.cat((output_eot[0][:lengths[0]+1], pt.sigmoid(output_eot)[0][:lengths[0]+1], eot_gt[0][:lengths[0]+1], ), dim=1))
  output_eot = pt.sigmoid(output_eot)
  pos_weight = pt.sum(eot_gt == 0)/pt.sum(eot_gt==1)
  neg_weight = 1
  eps = 1e-10
  # detach() for visualization
  eot_loss = pt.mean(-((pos_weight * eot_gt * pt.log(output_eot+eps)) + (neg_weight * (1-eot_gt)*pt.log(1-output_eot+eps))), dim=1).cpu().detach().numpy()
  # Thresholding the EOT to be class True/False
  threshold = 0.5
  output_eot = output_eot > threshold

  input_uv = input_uv.cpu().detach().numpy()
  output_eot = output_eot.cpu().detach().numpy()
  eot_gt = eot_gt.cpu().detach().numpy()
  lengths = lengths.cpu().detach().numpy()
  # marker_dict for contain the marker properties
  marker_dict_gt = dict(color='rgba(0, 0, 255, .5)', size=3)
  marker_dict_pred = dict(color='rgba(255, 0, 0, .5)', size=3)
  # Iterate to plot each trajectory
  for idx, i in enumerate(vis_idx):
    row_idx = idx+1
    cm_vis = confusion_matrix(y_pred=output_eot[i][:lengths[i]+1, :].reshape(-1) > 0.8, y_true=eot_gt[i][:lengths[i]+1, :].reshape(-1))
    tn, fp, fn, tp = cm_vis.ravel()
    # Plot the EOT
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]+1).reshape(-1,), y=output_eot[i][:lengths[i]+1, :].reshape(-1,), mode='markers+lines', marker=marker_dict_pred, name="{}-EOT Predicted [{}], EOTLoss = {:.3f}, [TP={}, FP={}, TN={}, FN={}, Precision={}, Recall={}]".format(flag, i, eot_loss[i][0], tp, fp, tn, fn, tp/(tp+fp), tp/(tp+fn))), row=row_idx, col=1)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]+1).reshape(-1,), y=eot_gt[i][:lengths[i]+1, :].reshape(-1,), mode='markers+lines', marker=marker_dict_gt, name="{}-Ground Truth EOT [{}]".format(flag, i)), row=row_idx, col=1)
    # Plot the input uv
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]).reshape(-1,), y=input_uv[i][:lengths[i], 0].reshape(-1,), mode='lines', marker=marker_dict_pred, name="Displacement - u"), row=row_idx, col=2)
    fig.add_trace(go.Scatter(x=np.arange(lengths[i]).reshape(-1,), y=input_uv[i][:lengths[i], 1].reshape(-1,), mode='lines', marker=marker_dict_gt, name="Displacement - v"), row=row_idx, col=2)

def EndOfTrajectoryLoss(output_eot, eot_gt, eot_startpos, mask, lengths):
  # Add the feature dimension using unsqueeze
  eot_gt = pt.unsqueeze(eot_gt, dim=2)
  mask = pt.unsqueeze(mask, dim=2)
  eot_startpos = pt.unsqueeze(eot_startpos, dim=2)
  # output_eot : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output_eot = pt.stack([pt.cat([eot_startpos[i], output_eot[i]]) for i in range(eot_startpos.shape[0])])
  output_eot *= mask
  eot_gt *= mask

  # Implement weighted BCE from scratch
  eot_gt = pt.cat(([eot_gt[i][:lengths[i]+1] for i in range(lengths.shape[0])]))
  output_eot = pt.sigmoid(pt.cat(([output_eot[i][:lengths[i]+1] for i in range(lengths.shape[0])])))
  # Weight of positive/negative classes for imbalanced class
  pos_weight = pt.sum(eot_gt == 0)/pt.sum(eot_gt==1)
  neg_weight = 1
  # Prevent of pt.log(-value)
  eps = 1e-10
  eot_loss = pt.mean(-((pos_weight * eot_gt * pt.log(output_eot + eps)) + (neg_weight * (1-eot_gt)*pt.log(1-output_eot + eps))))

  return eot_loss * 100

def evaluateModel(output_eot, eot_gt, eot_startpos, mask, lengths):
  # Add the feature dimension using unsqueeze
  eot_gt = pt.unsqueeze(eot_gt, dim=2)
  mask = pt.unsqueeze(mask, dim=2)
  eot_startpos = pt.unsqueeze(eot_startpos, dim=2)
  # output_eot : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output_eot = pt.stack([pt.cat([eot_startpos[i], output_eot[i]]) for i in range(eot_startpos.shape[0])])
  # Multiplied with mask
  output_eot *= mask
  eot_gt *= mask
  # Prediciton threshold
  threshold = 0.5

  # Each trajectory
  output_eot_each_traj = pt.sigmoid(output_eot.clone()).cpu().detach().numpy() > threshold
  eot_gt_each_traj = eot_gt.clone().cpu().detach().numpy()
  # Confusion matrix of each sample : Output from confusion_matrix.ravel() would be [TN, FP, FN, TP]
  cm_each_trajectory = [confusion_matrix(y_pred=output_eot_each_traj[i][:lengths[i]+1, :], y_true=eot_gt_each_traj[i][:lengths[i]+1, :]).ravel() for i in range(lengths.shape[0])]
  # Metrics of each sameple : precision_recall_fscore_support is Precision, Recall, Fbeta_score, support(#N of each label) with 2D array in format of [negative class, positive class]
  metrics_each_trajectory = [precision_recall_fscore_support(y_pred=output_eot_each_traj[i][:lengths[i]+1, :], y_true=eot_gt_each_traj[i][:lengths[i]+1, :]) for i in range(lengths.shape[0])]

  # Each batch
  eot_gt_batch = pt.cat(([eot_gt[i][:lengths[i]+1] for i in range(lengths.shape[0])]))
  output_eot_batch = pt.sigmoid(pt.cat(([output_eot[i][:lengths[i]+1] for i in range(lengths.shape[0])]))) > threshold
  eot_gt_batch = eot_gt_batch.cpu().detach().numpy()
  output_eot_batch = output_eot_batch.cpu().detach().numpy()

  # Confusion matrix of each batch : Output from confusion_matrix.ravel() would be [TN, FP, FN, TP]
  cm_batch = confusion_matrix(y_true=eot_gt_batch, y_pred=output_eot_batch).ravel()
  metrics_batch = precision_recall_fscore_support(y_pred=output_eot_batch, y_true=eot_gt_batch)
  tn, fp, fn, tp = cm_batch.ravel()
  print("Each Batch : TP = {}, FP = {}, TN = {}, FN = {}".format(tp, fp, tn, fn), end=', ')
  return np.array(cm_batch), np.array(cm_each_trajectory), np.array(metrics_each_trajectory)

def append_inference(inference_batch, lengths, eot_startpos, output_eot, eot_gt, mask):
  # Add the feature dimension using unsqueeze
  eot_gt = pt.unsqueeze(eot_gt, dim=2)
  mask = pt.unsqueeze(mask, dim=2)
  eot_startpos = pt.unsqueeze(eot_startpos, dim=2)
  # output_eot : concat with startpos and stack back to (batch_size, sequence_length+1, 1)
  output_eot = pt.stack([pt.cat([eot_startpos[i], output_eot[i]]) for i in range(eot_startpos.shape[0])])
  # Multiplied with mask
  output_eot *= mask
  eot_gt *= mask
  # Prediciton threshold
  threshold = 0.5

  # Each trajectory
  output_eot_each_traj = pt.sigmoid(output_eot.clone()).cpu().detach().numpy() > threshold
  eot_gt_each_traj = eot_gt.clone().cpu().detach().numpy()

  output_eot = output_eot.cpu().detach().numpy()
  inference_batch = [np.concatenate((inference_batch[i], output_eot_each_traj[i][:lengths[i]+1, :]), axis=1) for i in range(lengths.shape[0])]
  # Print out to check the EOT flag
  # for i in range(lengths[0]+1):
    # print('{} - {}'.format(inference_batch[0][i, -1], eot_gt_each_traj[0][i]))
  return inference_batch

def add_noise(input_trajectory, startpos, lengths):
  factor = np.random.uniform(low=0.6, high=0.95)
  if args.noise_sd is None:
    noise_sd = np.random.uniform(low=0.3, high=1.5)
  else:
    noise_sd = args.noise_sd
  input_trajectory = pt.cat((startpos[..., [0, 1]], input_trajectory), dim=1)
  input_trajectory = pt.cumsum(input_trajectory, dim=1)
  # plt.plot(np.diff(input_trajectory[0][:lengths[0]+1, 0].cpu().numpy()))
  # plt.plot(np.diff(input_trajectory[0][:lengths[0]+1, 1].cpu().numpy()))
  # plt.plot(np.diff(input_trajectory[0][:lengths[0]+1, -1].cpu().numpy()))
  noise_uv = pt.normal(mean=0.0, std=noise_sd, size=input_trajectory[..., :-1].shape).to(device)
  masking_noise = pt.nn.init.uniform_(pt.empty(input_trajectory[..., :-1].shape)).to(device) > np.random.rand(1)[0]
  n_noise = int(input_trajectory.shape[0] * factor)
  noise_idx = np.random.choice(a=input_trajectory.shape[0], size=(n_noise,), replace=False)
  input_trajectory[noise_idx, :, :] += noise_uv[noise_idx, :, :] * masking_noise[noise_idx, :, :]
  # plt.plot(np.diff(input_trajectory[0][:lengths[0]+1, 0].cpu().numpy()))
  # plt.plot(np.diff(input_trajectory[0][:lengths[0]+1, 1].cpu().numpy()))
  # plt.plot(np.diff(input_trajectory[0][:lengths[0]+1, -1].cpu().numpy()))
  # plt.show()
  # exit()
  input_trajectory = pt.tensor(np.diff(input_trajectory.cpu().numpy(), axis=1)).to(device)
  return input_trajectory

def predict(output_trajectory_test, output_trajectory_test_mask, output_trajectory_test_lengths, output_trajectory_test_startpos, output_trajectory_test_uv, input_trajectory_test, input_trajectory_test_mask, input_trajectory_test_lengths, input_trajectory_test_startpos, model, hidden, cell_state, inference_batch, visualize_trajectory_flag=True, visualization_path='./visualize_html/'):
  # Testing RNN/LSTM model
  # Initial hidden layer for the first RNN Cell
  # Test a model
  hidden = model.initHidden(batch_size=args.batch_size)
  cell_state = model.initCellState(batch_size=args.batch_size)
  # Evaluating mode
  model.eval()
  # Forward pass for validate a model
  input_trajectory_test_gt = input_trajectory_test.clone()
  if args.noise:
    input_trajectory_test = add_noise(input_trajectory=input_trajectory_test, startpos=input_trajectory_test_startpos, lengths=input_trajectory_test_lengths)
  output_test_eot, (_, _) = model(input_trajectory_test, hidden, cell_state, lengths=input_trajectory_test_lengths)
  # Detach for use hidden as a weights in next batch
  cell_state.detach()
  cell_state = cell_state.detach()
  hidden.detach()
  hidden = hidden.detach()

  # Stack the eot_pred
  inference_batch = append_inference(inference_batch=inference_batch, lengths=output_trajectory_test_lengths, eot_startpos=input_trajectory_test_startpos[..., -1], output_eot=output_test_eot.clone(), eot_gt=output_trajectory_test_uv[..., -1], mask=output_trajectory_test_mask[..., -1])

  # Calculate End of trajectory Loss
  test_eot_loss = EndOfTrajectoryLoss(output_eot=output_test_eot.clone(), eot_gt=output_trajectory_test_uv[..., -1], mask=output_trajectory_test_mask[..., -1], lengths=output_trajectory_test_lengths, eot_startpos=input_trajectory_test_startpos[..., -1])
  test_loss = test_eot_loss
  cm_batch, cm_each_trajectory, metrics_each_trajectory = evaluateModel(output_eot=output_test_eot.clone(), eot_gt=output_trajectory_test_uv[..., -1], mask=output_trajectory_test_mask[..., -1], lengths=output_trajectory_test_lengths, eot_startpos=input_trajectory_test_startpos[..., -1])

  # Get the accepted_trajectory in batch (Perfect trajectory with FP and FN == 0)
  try:
    accepted_trajectory = np.sum((np.logical_and(cm_each_trajectory[:, 1] == 0., cm_each_trajectory[:, 2] == 0.)))
  except IndexError:
    accepted_trajectory = 0
    print("[!!!] Prediction has only 1 or 0. Confusion matrix output only 1x1 matrix instead of 2x2")

  print('===> Test Loss (BCELogits Loss) : {:.3f}'.format(test_loss.item()))

  if visualize_trajectory_flag:
    make_visualize(input_trajectory_test=input_trajectory_test, output_test_eot=output_test_eot, output_trajectory_test_startpos=output_trajectory_test_startpos, input_trajectory_test_lengths=input_trajectory_test_lengths, output_trajectory_test_maks=output_trajectory_test_mask, visualization_path=visualization_path)
    input("\nContinue plotting...")

  # Calculate the EOT prediction accuracy, return the inference_batch stacked with the eot_pred
  return cm_batch, accepted_trajectory, inference_batch

def initialize_folder(path):
  if not os.path.exists(path):
      os.makedirs(path)

def collate_fn_padd(batch):
    '''
    Padding batch of variable length
    - The inference_batch will contain all other features (x, y, z, u, v, depth, eot_pred, eot_gt). This will enabled in save_eot and will write the file into numpy array to be used in other models
    '''
    padding_value = -10
    ## Get sequence lengths
    lengths = pt.tensor([trajectory[1:, :].shape[0] for trajectory in batch])
    # Input features : columns 4-5 contain u, v in screen space
    ## Padding 
    input_batch = [pt.Tensor(trajectory[1:, [3, 4]]) for trajectory in batch] # Mocap (3, 4, -1) = (u, v, end_of_trajectory)
    # input_batch = [pt.Tensor(trajectory[1:, [4, 5]]) for trajectory in batch] # Unity (4, 5, -2) = (u, v, end_of_trajectory)
    input_batch = pad_sequence(input_batch, batch_first=True, padding_value=padding_value)
    ## Retrieve initial position (u, v, depth)
    input_startpos = pt.stack([pt.Tensor(trajectory[0, [3, 4, -1]]) for trajectory in batch])  # (4, 5, 6, -2) = (u, v, end_of_trajectory)
    # input_startpos = pt.stack([pt.Tensor(trajectory[0, [4, 5, -2]]) for trajectory in batch]) # Unity
    input_startpos = pt.unsqueeze(input_startpos, dim=1)
    ## Compute mask
    input_mask = (input_batch != padding_value)

    # Output features : columns -2 cotain EOT ground truth
    ## Padding
    output_batch = [pt.Tensor(trajectory[1:, -1]) for trajectory in batch]
    output_batch = pad_sequence(output_batch, batch_first=True)
    ## Retrieve initial position
    output_startpos = pt.stack([pt.Tensor(trajectory[0, [-1]]) for trajectory in batch]) # Mocap
    # output_startpos = pt.stack([pt.Tensor(trajectory[0, [-2]]) for trajectory in batch]) # Unity
    output_startpos = pt.unsqueeze(output_startpos, dim=1)
    ## Retrieve the x, y, z in world space for compute the reprojection error (x', y', z' <===> x, y, z)
    output_uv = [pt.Tensor(trajectory[:, [3, 4, -1]]) for trajectory in batch]  # Mocap
    # output_uv = [pt.Tensor(trajectory[:, [4, 5, -2]]) for trajectory in batch]  # Unity
    output_uv = pad_sequence(output_uv, batch_first=True, padding_value=padding_value)
    ## Compute mask
    output_mask = (output_uv != padding_value)
    # Output mask is all trajectory points but input mask is get rid of startpos version
    ## Compute cummulative summation to form a trajectory from displacement every columns except the end_of_trajectory
    # print(output_xyz[..., :-1].shape, pt.unsqueeze(output_xyz[..., -1], dim=2).shape)
    output_uv = pt.cat((pt.cumsum(output_uv[..., :-1], dim=1), pt.unsqueeze(output_uv[..., -1], dim=2)), dim=2)

    # Inference features : thie is an output from inference that will take [x, y, z, u, v, depth, eot_gt, eot_pred] and write to npy file for futher prediction of 3D trajectory reconstruction
    # - eot_pred will append to the array once the inference is done.
    inference_batch = [pt.Tensor(trajectory[:, [0, 1, 2, 3, 4, 5, -1]]) for trajectory in batch]    # Mocap
    # inference_batch = [pt.Tensor(trajectory[:, [0, 1, 2, 3, 4, 5, -2]]) for trajectory in batch]    # Unity

    # print("Mask shape : ", mask.shape)
    return {'input':[input_batch, lengths, input_mask, input_startpos],
            'output':[output_batch, lengths, output_mask, output_startpos, output_uv],
            'inference':[inference_batch]}



def get_model(input_size, output_size, model_arch):
  if model_arch=='gru':
    model = GRU(input_size=input_size, output_size=output_size)
  elif model_arch=='bigru':
    model = BiGRU(input_size=input_size, output_size=output_size)
  elif model_arch=='bigru_residual_list':
    model = BiGRUResidualList(input_size=input_size, output_size=output_size)
  elif model_arch=='bigru_residual_add':
    model = BiGRUResidualAdd(input_size=input_size, output_size=output_size)
  elif model_arch=='lstm':
    model = LSTM(input_size=input_size, output_size=output_size)
  elif model_arch=='bigru':
    model = BiLSTM(input_size=input_size, output_size=output_size)

  return model

if __name__ == '__main__':
  print('[#]Training : Trajectory Estimation')
  # Argumentparser for input
  parser = argparse.ArgumentParser(description='Predict the 2D projectile')
  parser.add_argument('--dataset_test_path', dest='dataset_test_path', type=str, help='Path to testing set', required=True)
  parser.add_argument('--batch_size', dest='batch_size', type=int, help='Size of batch', default=50)
  parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, help='Type of trajectory(Rolling, Projectile, MagnusProjectile)', default='Projectile')
  parser.add_argument('--no_visualize', dest='visualize_trajectory_flag', help='No Visualize the trajectory', action='store_false')
  parser.add_argument('--visualize', dest='visualize_trajectory_flag', help='Visualize the trajectory', action='store_true')
  parser.add_argument('--pretrained_model_path', dest='pretrained_model_path', type=str, help='Path to load a trained model checkpoint', required=True)
  parser.add_argument('--visualization_path', dest='visualization_path', type=str, help='Path to visualization directory', default='./visualize_html/')
  parser.add_argument('--threshold', dest='threshold', type=float, help='Provide the error threshold of reconstructed trajectory', default=0.8)
  parser.add_argument('--cuda_device_num', dest='cuda_device_num', type=int, help='Provide cuda device number', default=0)
  parser.add_argument('--model_arch', dest='model_arch', type=str, help='Input the model architecture(lstm, bilstm, gru, bigru)', required=True)
  parser.add_argument('--save_eot_path', dest='save_eot_path', type=str, default=None, help='Path to save the inference output , if this is None so it will not save')
  parser.add_argument('--noise', dest='noise', help='Noise on the fly', action='store_true')
  parser.add_argument('--no_noise', dest='noise', help='Noise on the fly', action='store_false')
  parser.add_argument('--noise_sd', dest='noise_sd', help='Std. of noise', type=float, default=None)
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
    # 2.RNN/LSTM modelOSAI
    # 3.Unpack the packed
    unpacked = pt.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, padding_value=-1)
    print("Unpacked equality : ", pt.eq(batch['input'][0], unpacked[0]).all())
    print("===============================================================================================================================================================")

  # Model definition
  n_output = 1 # Contain the depth information of the trajectory and the end_of_trajectory flag
  n_input = 2 # Contain following this trajectory parameters (u, v, end_of_trajectory) position from tracking
  print('[#]Model Architecture')
  model = get_model(input_size=n_input, output_size=n_output, model_arch=args.model_arch)
  if args.pretrained_model_path is None:
    print('===>No pre-trained model to load')
    print('EXIT...')
    exit()
  else:
    print('===>Load trained model')
    model.load_state_dict(pt.load(args.pretrained_model_path, map_location=device)['model'])
  model = model.to(device)
  print(model)

  hidden = model.initHidden(batch_size=args.batch_size)
  cell_state = model.initCellState(batch_size=args.batch_size)
  # Test a model iterate over dataloader to get each batch and pass to predict function
  n_trajectory = 0
  n_accepted_trajectory = 0 # Perfect trajectory without any misclassification (No FN, FP)
  cm_entire_dataset = np.zeros(4)
  inference_output = []
  # Testing a model iterate over dataloader to get each batch and pass to predict function
  for batch_idx, batch_test in enumerate(trajectory_test_dataloader):
    # testing set (Each index in batch_test came from the collate_fn_padd)
    input_trajectory_test = batch_test['input'][0].to(device)
    input_trajectory_test_lengths = batch_test['input'][1].to(device)
    input_trajectory_test_mask = batch_test['input'][2].to(device)
    input_trajectory_test_startpos = batch_test['input'][3].to(device)
    output_trajectory_test = batch_test['output'][0].to(device)
    output_trajectory_test_lengths = batch_test['output'][1].to(device)
    output_trajectory_test_mask = batch_test['output'][2].to(device)
    output_trajectory_test_startpos = batch_test['output'][3].to(device)
    output_trajectory_test_uv = batch_test['output'][4].to(device)
    inference_batch = batch_test['inference'][0]

    # Call function to test
    cm_batch, accepted_trajectory, inference_batch  = predict(output_trajectory_test=output_trajectory_test, output_trajectory_test_mask=output_trajectory_test_mask,
                                                              output_trajectory_test_lengths=output_trajectory_test_lengths, output_trajectory_test_startpos=output_trajectory_test_startpos,
                                                              output_trajectory_test_uv=output_trajectory_test_uv,
                                                              input_trajectory_test=input_trajectory_test, input_trajectory_test_mask = input_trajectory_test_mask,
                                                              input_trajectory_test_lengths=input_trajectory_test_lengths, input_trajectory_test_startpos=input_trajectory_test_startpos,
                                                              model=model, hidden=hidden, cell_state=cell_state, visualize_trajectory_flag=args.visualize_trajectory_flag,
                                                              inference_batch=inference_batch)

    # Append the inference results into inference output list in shape of (n_trajectory, sequence_length, n_features(x, y, z, u, v, depth, eot_gt, eot_pred))
    print("#N trajectory : ", np.array(inference_batch).shape)
    print("Example trajectory : ", np.array(inference_batch)[0].shape)
    inference_output.append(np.array(inference_batch))

    cm_entire_dataset += np.array(cm_batch)
    n_accepted_trajectory += accepted_trajectory
    n_trajectory += input_trajectory_test_lengths.shape[0]


  inference_output = np.array([inference_output[i][j] for i in range(len(inference_output)) for j in range(len(inference_output[i]))])
  # print(inference_output)
  print("[#]Saving the inference output to .npy format... : ", np.array(inference_output).shape)
  # Saving the npy inference output : shape is (x, y, z, u, v, depth, eot_gt, eot_pred)
  if args.save_eot_path is not None:
    print("===>SHAPE : ", inference_output.shape)
    # print(inference_output[0].shape)
    with open(args.save_eot_path, 'wb') as outfile:
      np.save(outfile, inference_output)

  # Calculate the metrics to summary
  tn, fp, fn, tp = cm_entire_dataset.ravel()
  acc = (tp + tn)/(tn+fp+fn+tp)
  recall = tp/(tp+fn)   # How model can retrieve the all actual positive
  precision = tp/(tp+fp)    # How correct of prediciton from all prediciton as positive
  f1_score = 2 * (recall*precision)/(precision+recall)

  print("="*100)
  print("[#]Summary")
  print("Entire Dataset : TP = {}, FP = {}, TN = {}, FN = {}".format(tp, fp, tn, fn))
  print("===>Accuracy = {}".format(acc))
  print("===>Recall = {}".format(recall))
  print("===>Precision = {}".format(precision))
  print("===>F1-score = {}".format(f1_score))
  print("#N accepted trajectory [Perfect trajectory(No FP and FN)] : {} from {}".format(n_accepted_trajectory, n_trajectory))
  print("="*100)
  print("[#] Done")
