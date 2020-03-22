from __future__ import print_function
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import glob
import os
import argparse
from tqdm import tqdm
from rnn_model import RNN
from lstm_model import LSTM
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from dataloader import TrajectoryDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import PIL.Image
import io

def visualize_trajectory(output, trajectory, writer, flag='predict', n_vis=5):
  # Reshape to (#N_Trajectory, Trajectory_length, (x, y))
  if flag=='train':
    output = output.reshape(trajectory.shape[0], trajectory.shape[1], 2)
  vis_idx = np.random.randint(low=0, high=trajectory.shape[0], size=(n_vis))
  traj_img_list = []
  for i in vis_idx:
    img_buffer = io.BytesIO()
    plt.title('Trajectory Estimation')
    plt.cla()
    plt.scatter(output[i][..., 0], output[i][..., 1], marker='^', label='Estimated')
    plt.scatter(trajectory[i][..., 0], trajectory[i][..., 1], marker='*', label='Ground Truth')
    plt.legend()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    traj_img = PIL.Image.open(img_buffer)
    traj_img = ToTensor()(traj_img)
    traj_img_list.append(np.array(traj_img[:3, ...]))   # Remove 4th channel which is transparency for PNG format
  return np.array(traj_img_list)

def MSELoss(output, trajectory_gt, mask, delmask=True):
  mse_loss = pt.sum(((trajectory_gt - output)**2) * mask) / pt.sum(mask)
  return mse_loss

def train(output_trajectory_train, output_trajectory_train_mask, input_trajectory_train, input_trajectory_train_mask, model, output_trajectory_val, output_trajectory_val_mask, input_trajectory_val, input_trajectory_val_mask, hidden, cell_state, visualize_trajectory_flag=True, writer=None, min_val_loss=2e10, model_checkpoint_path='./model/'):
  # Training RNN/LSTM model 
  # Run over each example
  # trajectory_train = trajectory_train path with shape (n_trajectory_train, 2) ===> All 2 features are (x0, y0) ... (xn, yn) ;until yn == 0
  # initial_condition_train = Initial conditon with shape (n_trajectory_train, 6) ===> All 6 features are (x, y, angle, velocity, g, timestep)
  # Define models parameters
  learning_rate = 0.001
  n_epochs = 300
  optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate)
  # Initial hidden layer for the first RNN Cell
  model.train()
  # Train a model
  for epoch in range(1, n_epochs+1):
    optimizer.zero_grad() # Clear existing gradients from previous epoch
    # Forward PASSING
    # Forward pass for training a model
    output_train, (hidden, cell_state) = model(input_trajectory_train, hidden, cell_state)
    output_train = output_train.view((output_trajectory_train.size()[0], output_trajectory_train.size()[1], output_trajectory_train.size()[2]))
    output_train = pt.mul(output_train, output_trajectory_train_mask)
    # Forward pass for validate a model
    output_val, (_, _) = model(input_trajectory_val, hidden, cell_state)
    output_val = output_val.view((output_trajectory_val.size()[0], output_trajectory_val.size()[1], output_trajectory_val.size()[2]))
    output_val = pt.mul(output_val, output_trajectory_val_mask)
    # Detach for use hidden as a weights in next batch
    cell_state.detach()
    cell_state = cell_state.detach()
    hidden.detach()
    hidden = hidden.detach()

    # Apply Cummulative summation for transfer the displacement to the x, y coordinate
    # Calculate loss by displacement
    # train_loss = loss_fn(output_train, trajectory_train.float())
    # val_loss = loss_fn(output_val, trajectory_val.float())
    train_loss = MSELoss(output_train, output_trajectory_train, output_trajectory_train_mask)
    val_loss = MSELoss(output_val, output_trajectory_val, output_trajectory_val_mask)

    train_loss.backward() # Perform a backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly to the gradients

    if epoch%10 == 0:
      print('Epoch : {}/{}.........'.format(epoch, n_epochs), end='')
      print('Train Loss : {:.3f}'.format(train_loss.item()), end=', ')
      print('Val Loss : {:.3f}'.format(val_loss.item()))
      writer.add_scalars('Loss/', {'Training loss':train_loss.item(),
                                  'Validation loss':val_loss.item()}, epoch)
      # writer.add_scalar('Loss/val_loss', val_loss.item(), epoch)
      if visualize_trajectory_flag == True:
        traj_train_img = visualize_trajectory(pt.cumsum(output_train, dim=1).cpu().detach().numpy(), pt.cumsum(trajectory_train, dim=1).cpu().detach().numpy(), writer=writer, flag='train')
        writer.add_image('Training set : Trajectory Estimation', traj_train_img, epoch, dataformats='NCHW')
        traj_val_img = visualize_trajectory(pt.cumsum(output_val, dim=1).cpu().detach().numpy(), pt.cumsum(trajectory_val, dim=1).cpu().detach().numpy(), writer=writer, flag='train')
        writer.add_image('Validation set : Trajectory Estimation', traj_val_img, epoch, dataformats='NCHW')
      # Save model checkpoint
      if min_val_loss > val_loss:
        print('[#]Saving a model checkpoint')
        min_val_loss = val_loss
        pt.save(model.state_dict(), args.model_checkpoint_path)

  return min_val_loss, hidden, cell_state

def predict(trajectory_gt, initial_condition_gt, model, visualize_trajectory_flag=True, writer=None):
  trajectory_gt_unpadded, initial_condition_gt_unpadded = np.copy(unpadded_tensor(np.copy(trajectory_gt.cpu().detach().clone().numpy()), np.copy(initial_condition_gt.cpu().detach().clone().numpy())))
  # Trajectory size = (#n trajectory, #seq_length, #n_output_coordinates)
  output_pred = np.copy(initial_condition_gt_unpadded)
  # output_pred = np.insert(output_pred, output_pred.shape[1], values=[-10, -10], axis=1)
  # Initial condition size = (#n trajectory, #seq_length, #n_input_coordinates)delta in vector space
  initial_condition_pred = np.copy(initial_condition_gt_unpadded)
  # Loop over every trajectory
  loss_fn = pt.nn.MSELoss()
  n_prior_point = 15
  model.eval()
  with pt.no_grad():
    for i in tqdm(range(trajectory_gt_unpadded.shape[0]), desc='Prediction Trajectory'):
      # Loop over length of the trajectory
      batch_size=1
      hidden = model.initHidden(batch_size=batch_size)
      cell_state = model.initCellState(batch_size=batch_size)
      # print(trajectory_gt_unpadded[i].shape)
      # print(initial_condition_gt_unpadded[i].shape)
      # print(output_pred[i].shape)
      for j in range(n_prior_point, trajectory_gt_unpadded[i].shape[0]):
        # print('All points {} : From {} to {}'.format(trajectory_gt_unpadded[i].shape[0], j-n_prior_point, j))
        # Init the initial_condition_pred from initial_condition_gt for the beginning of the trajectory
        # Make a prediction
        output, (hidden, cell_state) = model(pt.from_numpy(output_pred[i][j-n_prior_point:j].reshape(1, n_prior_point, 2)).cuda().float(), hidden, cell_state)
        try:
          output_pred[i][j] = np.copy(output[-1][:].cpu().detach().clone().numpy())
        except IndexError:
          output_pred[i] = np.vstack((output_pred[i], output[-1][:].cpu().detach().clone().numpy()))

        # print("LAST : ", model(initial_condition_pred[i][j-n_prior_point:j].reshape(1, n_prior_point, 2).float())[0][-1][:])
        # print("FIRST : ", model(initial_condition_pred[i][j-n_prior_point:j].reshape(1, n_prior_point, 2).float())[0][:][:])
        # Change the current point from predicted point not the ground truth point
        # initial_condition_pred[i][j][0] = np.copy(output_pred[i][j][0])
        # initial_condition_pred[i][j][1] = np.copy(output_pred[i][j][1])
        # print('=============={}=============='.format(j))
        # print('Input : ', output_pred[i][j-n_prior_point:j])
        # print('Prediction : ', output_pred[i][j])
        # print('Ground Truth : ', trajectory_gt[i][j-1])
      # print('Loss : ', loss_fn(pt.from_numpy(output_pred[i][:]).cuda().float(), pt.from_numpy(trajectory_gt_unpadded[i][:]).cuda().float()))
    if visualize_trajectory_flag == True:
      output_pred = np.array([np.cumsum(output_pred[i], axis=0)  for i in range(len(output_pred))])
      trajectory_gt_unpadded = np.array([np.cumsum(trajectory_gt_unpadded[i], axis=0)  for i in range(len(trajectory_gt_unpadded))])
      traj_pred_img = visualize_trajectory(output_pred, trajectory_gt_unpadded, writer=writer, n_vis=32)
      writer.add_image('Testing set : Trajectory Estimation', traj_pred_img, dataformats='NCHW')

def collate_fn_padd(batch):
    '''
    Padding batch of variable length
    '''
    ## Get sequence lengths
    lengths = pt.tensor([trajectory[1:, :].shape[0] for trajectory in batch])
    # Input features : columns 4-5 contain u, v in screen space
    ## Padding 
    input_batch = [pt.Tensor(trajectory[1:, 4:6]) for trajectory in batch]
    input_batch = pt.nn.utils.rnn.pad_sequence(input_batch, batch_first=True)
    ## Retrieve initial position
    input_initpos = pt.stack([pt.Tensor(trajectory[0, 4:6]) for trajectory in batch])
    ## Compute mask
    input_mask = (input_batch != 0)

    # Output features : columns 0-2 cotain x, y, z in world space
    ## Padding
    output_batch = [pt.Tensor(trajectory[1:, :3]) for trajectory in batch]
    output_batch = pt.nn.utils.rnn.pad_sequence(output_batch, batch_first=True)
    ## Retrieve initial position
    output_initpos = pt.stack([pt.Tensor(trajectory[0, :3]) for trajectory in batch])
    ## Compute mask
    output_mask = (output_batch != 0)

    # print("Mask shape : ", mask.shape)
    return {'input':[input_batch, lengths, input_mask, input_initpos],
            'output':[output_batch, lengths, output_mask, output_initpos]}

if __name__ == '__main__':
  print('[#]Trajectory Estimation')
  # Argumentparser for input
  parser = argparse.ArgumentParser(description='Predict the 2D projectile')
  parser.add_argument('--dataset_path', dest='dataset_path', type=str, help='Path to dataset', required=True)
  parser.add_argument('--batch_size', dest='batch_size', type=int, help='Size of batch', default=50)
  parser.add_argument('--trajectory_type', dest='trajectory_type', type=str, help='Type of trajectory(Rolling, Projectile, MagnusProjectile)', default='Projectile')
  parser.add_argument('--visualize_trajectory_flag', dest='visualize_trajectory_flag', type=bool, help='Visualize the trajectory', default=False)
  parser.add_argument('--model_checkpoint_path', dest='model_checkpoint_path', type=str, help='Path to save a model checkpoint', required=True)
  parser.add_argument('--model_path', dest='model_path', type=str, help='Path to load a trained model checkpoint', default=None)
  args = parser.parse_args()

  # Create Datasetloader
  trajectory_dataset = TrajectoryDataset(dataset_path=args.dataset_path, trajectory_type=args.trajectory_type)
  # validation_set = RandomSampler(trajectory_dataset, replacement=False)
  # print(validation_set)
  trajectory_dataloader = DataLoader(trajectory_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn_padd, pin_memory=True, drop_last=True)

  # Dataset format
  # Trajectory path : (x0, y0) ... (xn, yn)
  print("======================================================Summary Batch (batch_size = {})=========================================================================".format(args.batch_size))
  for key, batch in enumerate(trajectory_dataloader):
    print("Input batch [{}] : batch={}, lengths={}, mask={}, initial position={}".format(key, batch['input'][0].shape, batch['input'][1].shape, batch['input'][2].shape, batch['input'][3].shape))
    print("Output batch [{}] : batch={}, lengths={}, mask={}, initial position={}".format(key, batch['output'][0].shape, batch['output'][1].shape, batch['output'][2].shape, batch['output'][3].shape))
    # Test RNN/LSTM Step
    # 1.Pack the padded
    packed = pt.nn.utils.rnn.pack_padded_sequence(batch['input'][0], batch_first=True, lengths=batch['input'][1], enforce_sorted=False)
    # 2.RNN/LSTM model
    # 3.Unpack the packed
    unpacked = pt.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
    print("Unpacked equality : ", pt.eq(batch['input'][0], unpacked[0]).all())
    print("===============================================================================================================================================================")

  # Model definition
  hidden_dim = 32
  n_output = 3 # Contain the depth information of the trajectory
  n_input = 2 # Contain following this trajectory parameters (u, v) position from tracking
  min_val_loss = 2e10
  # GPU initialization
  if pt.cuda.is_available():
    device = pt.device('cuda')
    print('[%]GPU Enabled')
  else:
    device = pt.device('cpu')
    print('[%]GPU Disabled, CPU Enabled')

  print('[#]Model Architecture')
  if args.model_path is None:
    # Create a model
    print('===>No trained model')
    rnn_model = LSTM(input_size=n_input, output_size=n_output, hidden_dim=hidden_dim, n_layers=8)
  else:
    print('===>Load trained model')
    rnn_model = LSTM(input_size=n_input_, output_size=n_output, hidden_dim=hidden_dim, n_layers=8)
    rnn_model.load_state_dict(pt.load(args.model_path))
  rnn_model = rnn_model.to(device)
  print(rnn_model)
  # Initial writer for tensorboard
  writer = SummaryWriter('trajectory_tensorboard/{}'.format(args.dataset_path))

  # batch_size = trajectory[0].size()[0]
  hidden = rnn_model.initHidden(batch_size=args.batch_size)
  cell_state = rnn_model.initCellState(batch_size=args.batch_size)
  # Training a model
  for batch_idx, batch in enumerate(trajectory_dataloader):
    # Training set
    input_trajectory_train = batch['input'][0].to(device)
    input_trajectory_train_mask = batch['input'][2].to(device)
    output_trajectory_train = batch['output'][0].to(device)
    output_trajectory_train_mask = batch['output'][2].to(device)
    # Validation set
    input_trajectory_val = batch['input'][0].to(device)
    input_trajectory_val_mask = batch['input'][2].to(device)
    output_trajectory_val = batch['output'][0].to(device)
    output_trajectory_val_mask = batch['output'][2].to(device)
    # Call function to train
    min_val_loss, hidden, cell_state = train(output_trajectory_train=output_trajectory_train, output_trajectory_train_mask=output_trajectory_train_mask,
                                             input_trajectory_train=input_trajectory_train, input_trajectory_train_mask = input_trajectory_train_mask,
                                             output_trajectory_val=output_trajectory_val, output_trajectory_val_mask=output_trajectory_val_mask,
                                             input_trajectory_val=input_trajectory_val, input_trajectory_val_mask=input_trajectory_val_mask,
                                             model=rnn_model, hidden=hidden, cell_state=cell_state, visualize_trajectory_flag=args.visualize_trajectory_flag,
                                             writer=writer, min_val_loss=min_val_loss, model_checkpoint_path=args.model_checkpoint_path)

  # Prediction on test set
  predict(trajectory_gt=trajectory[test_fold][:], initial_condition_gt=initial_condition[test_fold][:], model=rnn_model, visualize_trajectory_flag=args.visualize_trajectory_flag, writer=writer)
