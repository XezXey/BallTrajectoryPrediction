import numpy as np
import json
import argparse
import pandas as pd
import json
import tqdm
import glob
import os
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
<<<<<<< HEAD

def get_selected_cols():
  # Flag/Extra features columns
  features_cols = []
  if 'eot' in args.selected_features:
    features_cols.append('end_of_trajectory')
  if 'og' in args.selected_features:
    features_cols.append('on_ground_flag')
  if 'f_rad' in args.selected_features:
    features_cols.append('force_angle_rad')
  if 'f_sin' in args.selected_features:
    features_cols.append('force_sine')
  if 'f_cos' in args.selected_features:
    features_cols.append('force_cosine')

  # Position columns
  position_cols = ['ball_world_x', 'ball_world_y', 'ball_world_z']
  for axis in ['x', 'y', 'z']:
    position_cols.append('ball_{}_{}_{}'.format(args.selected_space, args.selected_cams, axis))

  print("Selected features columns : ", features_cols)
  print("Selected position columns : ", position_cols)
  return features_cols, position_cols


def computeDisplacement(trajectory_split, trajectory_type):
  # Compute the displacement
  features_cols, position_cols = get_selected_cols()
  drop_cols = ["outside_flag", "trajectory_type", "t"] + features_cols
=======

def computeDisplacement(trajectory_split, trajectory_type):
  # Compute the displacement
  drop_cols = ["end_of_trajectory", "on_ground_flag", "add_force_flag", "outside_flag", "trajectory_type", "t"]
>>>>>>> 089ae3e542e8ea85ecac4da4d9d04183a5910226
  trajectory_npy = trajectory_split.copy()
  for traj_type in trajectory_type:
    print("Average of Y-axis Trajectory : ", np.mean([np.mean(trajectory_split[traj_type][i]['ball_world_y'].values) for i in range(len(trajectory_split[traj_type]))]))
    # Keep the first point as a starting point for performing a cumsum to retrieve whole trajectory 
    # First vstack(extend rows) with (First row, np.diff() of the rest)
    # Second hstack(extend columns) with (All columns, ['end_of_trajectory'] column) 
    # print(trajectory_split[traj_type][0].iloc[1:, :].shape)
    # print(trajectory_split[traj_type][0][['on_ground_flag']].iloc[1:, :].values.shape)
    # print(trajectory_split[traj_type][0].loc[:, ['end_of_trajectory']].values.shape)
    # print(np.zeros((1, 1)).shape)
    # print(np.concatenate((trajectory_split[traj_type][0].loc[1:, ['on_ground_flag']].values, np.ones((1, 1)))).shape)
    # exit()
<<<<<<< HEAD
    trajectory_npy[traj_type] = [np.hstack((np.vstack((trajectory_split[traj_type][i][position_cols].iloc[0].values,
                                                       np.diff(trajectory_split[traj_type][i][position_cols].values, axis=0))),
                                            trajectory_split[traj_type][i][features_cols].values,))
                                 for i in range(len(trajectory_split[traj_type]))]
=======
    if args.on_ground_flag:
      trajectory_npy[traj_type] = [np.hstack((np.vstack((trajectory_split[traj_type][i].drop(drop_cols, axis=1).iloc[0, :].values,
                                                         np.diff(trajectory_split[traj_type][i].drop(drop_cols, axis=1).values, axis=0))),
                                              # trajectory_split[traj_type][i].loc[:, ['end_of_trajectory']].values.astype(np.int64),
                                              trajectory_split[traj_type][i].loc[:, ['end_of_trajectory', 'on_ground_flag']].values.astype(np.int64),
                                              # np.concatenate((trajectory_split[traj_type][i][['on_ground_flag']].iloc[1:, :].values, np.ones((1, 1))))
                                              # np.concatenate((trajectory_split[traj_type][i][['on_ground_flag']].values))
                                              )) for i in range(len(trajectory_split[traj_type]))]
    else :
      trajectory_npy[traj_type] = [np.hstack((np.vstack((trajectory_split[traj_type][i].drop(drop_cols, axis=1).iloc[0, :].values,
                                                         np.diff(trajectory_split[traj_type][i].drop(drop_cols, axis=1).values, axis=0))),
                                              trajectory_split[traj_type][i].loc[:, ['end_of_trajectory']].values.astype(np.int64))) for i in range(len(trajectory_split[traj_type]))]
>>>>>>> 089ae3e542e8ea85ecac4da4d9d04183a5910226
    # Cast to ndarray (Bunch of trajectory)
    temp = trajectory_npy[traj_type][0]
    plt.plot(np.arange(temp.shape[0]-1), temp[1:, [3]], '-or')
    plt.plot(np.arange(temp.shape[0]-1), temp[1:, [4]], '-og')
    plt.plot(np.arange(temp.shape[0]-1), temp[1:, [6]], '-ob')
    plt.show()
    trajectory_npy[traj_type] = np.array([trajectory_npy[traj_type][i] for i in range(len(trajectory_npy[traj_type]))])
    # Remove some dataset that goes below the ground (Error from unity)
    trajectory_npy[traj_type] = remove_below_ground_trajectory(trajectory=trajectory_npy[traj_type], traj_type=traj_type)
  return trajectory_npy

def remove_below_ground_trajectory(trajectory, traj_type):
  # Loop over the trajectory to remove any trajectory that goes below the ground
  # Also remove the trajectory that is outside the field and droping to the ground
  count=0
  eps = np.finfo(float).eps
  remove_idx = []
  for idx in range(trajectory.shape[0]):
    traj_cumsum_temp = np.cumsum(trajectory[idx][:, :], axis=0)
    if (np.any(traj_cumsum_temp[:, 1] <= -0.1)):
      remove_idx.append(idx)
      count+=1
<<<<<<< HEAD
  print("\n{}===>Remove the below ground trajectory : {} from {} at {}".format(traj_type, count, trajectory.shape[0], remove_idx))
=======
  print("\n{}===>Remove the below ground trajectory : {} at {}".format(traj_type, count, remove_idx))
>>>>>>> 089ae3e542e8ea85ecac4da4d9d04183a5910226
  trajectory = np.delete(trajectory.copy(), obj=remove_idx)
  return trajectory

def visualize_noise(trajectory):
  # Function for visualzie the trajectory given original dataset 
  marker_dict_u = dict(color='rgba(255, 0, 0, 0.2)', size=4)
  marker_dict_v = dict(color='rgba(0, 255, 0, 0.4)', size=4)
  marker_dict_depth = dict(color='rgba(0, 0, 255, 0.4)', size=4)
  marker_dict_uv = dict(color='rgba(255, 0, 0, 0.4)', size=4)
  marker_dict_xyz = dict(color='rgba(0, 0, 255, 0.4)', size=4)
  fig = make_subplots(rows=2, cols=2, specs=[[{'type':'scatter3d'}, {'type':'scatter'}], [{'type':'scatter'}, {'type':'scatter'}]])
  fig.add_trace(go.Scatter3d(x=trajectory.iloc[:, 0], y=trajectory.iloc[:, 1], z=trajectory.iloc[:, 2], marker=marker_dict_xyz, mode='markers'), row=1, col=1)
  fig.add_trace(go.Scatter(x=trajectory.iloc[:, 4], y=trajectory.iloc[:, 5], marker=marker_dict_uv, mode='markers'), row=1, col=2)
  fig.add_trace(go.Scatter(x=np.arange(np.diff(trajectory.iloc[:, 4]).shape[0]), y=np.diff(trajectory.iloc[:, 4]), marker=marker_dict_u, mode='lines'), row=2, col=1)
  fig.add_trace(go.Scatter(x=np.arange(np.diff(trajectory.iloc[:, 5]).shape[0]), y=np.diff(trajectory.iloc[:, 5]), marker=marker_dict_v, mode='lines'), row=2, col=1)
  fig.add_trace(go.Scatter(x=np.arange(np.diff(trajectory.iloc[:, 6]).shape[0]), y=np.diff(trajectory.iloc[:, 6]), marker=marker_dict_depth, mode='lines'), row=2, col=1)
  fig.show()

def worldToScreen(world, camera_config):
  projectionMatrix = camera_config['projectionMatrix'].copy()
  worldToCameraMatrix = camera_config['worldToCameraMatrix'].copy()
  width = camera_config['width']
  height = camera_config['height']
  # Remove the clipping stuff from projectionMatrix
  projectionMatrix[2, :] = projectionMatrix[3, :]
  projectionMatrix[3, :] = np.array([0, 0, 0, 1])
  # world space
  temp = np.ones((world.shape[0], world.shape[1]+1))
  temp[:, :-1] = world
  # NDC space
  ndc_space = (temp @ (projectionMatrix @ worldToCameraMatrix).T)
  if (np.all(ndc_space[:, -1] == 0)):
    # Points is exaclty on the camera focus, screen point is undefined, unity handles this by returning (0, 0, 0)
    return np.zeros(shape=ndc_space[:, :3].shape)
  else:
    screen_space = ndc_space.copy()
    screen_space[:, 0] = (ndc_space[:, 0]/ndc_space[:, 2] + 1) * 0.5 * width
    screen_space[:, 1] = (ndc_space[:, 1]/ndc_space[:, 2] + 1) * 0.5 * height
    return screen_space

def add_noise(trajectory_split, trajectory_type, camera_config):
  for traj_type in trajectory_type:
    # Visualize before an effect of noise
    if args.vis_noise:
      vis_idx = np.random.randint(0, len(trajectory_split[traj_type]))
      visualize_noise(trajectory_split[traj_type][vis_idx])
    # Get the noise offset
    noise = [np.random.normal(loc=0.0, scale=47e-3, size=trajectory_split[traj_type][i].iloc[:, :3].shape) for i in range(len(trajectory_split[traj_type]))]
    if args.masking:
      mask = [np.random.random(size=trajectory_split[traj_type][i].iloc[:, :3].shape) < 0.20 for i in range(len(trajectory_split[traj_type]))]
      noise = [noise[i] * mask[i] for i in range(len(trajectory_split[traj_type]))]

    # Apply noise to world space
    noisy_world = [trajectory_split[traj_type][i].iloc[:, :3].copy().values + noise[i]  for i in range(len(trajectory_split[traj_type]))]
    # Get the noisy screen space
    noisy_uv = [worldToScreen(world=noisy_world[i], camera_config=camera_config) for i in range(len(trajectory_split[traj_type]))]
    # Assign it to original trajectory_split variable
    for i in tqdm.tqdm(range(len(trajectory_split[traj_type])), desc="Replace the nosied pixel space"):
      # Replace only screen space columns
      trajectory_split[traj_type][i].iloc[:, 4:6] = noisy_uv[i][:, :2]
      trajectory_split[traj_type][i].iloc[:, :3] = noisy_world[i][:, :3]
    # Visualize after an effect of noise
    if args.vis_noise:
      temp_plot_trajectory = trajectory_split[traj_type][vis_idx].copy()
      temp_plot_trajectory.iloc[:, :3] = noisy_world[vis_idx]
      visualize_noise(temp_plot_trajectory)

  return trajectory_split

def split_by_flag(trajectory_df, trajectory_type, num_continuous_trajectory, timelag, flag='add_force_flag', force_zero_ground_flag=False, random_sampling_mode=False):
  trajectory_split = trajectory_df
  for traj_type in trajectory_type:
    if traj_type=='Rolling' and force_zero_ground_flag is True:
      trajectory_df[traj_type].iloc[:, 1] = trajectory_df[traj_type].iloc[:, 1] * 0.0
    elif force_zero_ground_flag is True:    # Notice that this will effect to the outside trajectory(The pitch will getting bigger since instead reject outside trajectory by filter with -value, we force all to zero)
      zero_threshold = 1e-3
      mask_y = trajectory_df[traj_type].iloc[:, 1].values > zero_threshold
      # print("Before froce zero ground")
      trajectory_df[traj_type].iloc[:, 1] = trajectory_df[traj_type].iloc[:, 1] * mask_y
      # print("After froce zero ground")

<<<<<<< HEAD
    trajectory_df[traj_type] = trajectory_df[traj_type].replace({True:1, False:0})
=======
    trajectory_df[traj_type] = trajectory_df[traj_type].replace({"True":True, "False":False})
>>>>>>> 089ae3e542e8ea85ecac4da4d9d04183a5910226
    # Split each dataframe by using the flag == True as an index of starting point
    index_split_by_flag = list(trajectory_df[traj_type].loc[trajectory_df[traj_type][flag] == True].index)[0:-1] # remove the first trajectory and the last trajectory
    # Store splitted dataframe in list (Not use the first and last trajectory : First one can be bug if the ball is not on the 100% ground, Last one is the the complete trajectory)
    if random_sampling_mode:
      trajectory_split[traj_type] = generate_random_num_continuous_trajectory(trajectory_df=trajectory_df, index_split_by_flag=index_split_by_flag, num_continuous_trajectory=num_continuous_trajectory, traj_type=traj_type, timelag=timelag)
    else:
      trajectory_split[traj_type] = generate_constant_num_continuous_trajectory(trajectory_df=trajectory_df, index_split_by_flag=index_split_by_flag, num_continuous_trajectory=num_continuous_trajectory, traj_type=traj_type, timelag=timelag)
    trajectory_split[traj_type] = get_end_of_trajectory_flag(trajectory_split=trajectory_split[traj_type], timelag=timelag)
<<<<<<< HEAD
  return trajectory_split

def get_end_of_trajectory_flag(trajectory_split, timelag):
  for i in range(len(trajectory_split)):
    unflip_add_force = trajectory_split[i]['add_force_flag'].values # Get the unflip add_force_flag value columns on the trajectory i-th index
    index_split_by_add_force_flag = list(trajectory_split[i].loc[trajectory_split[i]['add_force_flag'] == True].index)[:] # remove the first trajectory and the last trajectory
    index_split_by_add_force_flag.append(len(unflip_add_force) + index_split_by_add_force_flag[0]) # Index of each trajectory
    index_split_by_add_force_flag = np.array(index_split_by_add_force_flag) - index_split_by_add_force_flag[0]  # Re-index every row to start from 0
    flipped_add_force = [np.flip(unflip_add_force[index_split_by_add_force_flag[j]:index_split_by_add_force_flag[j+1]]) for j in range(len(index_split_by_add_force_flag)-1)]   # Get each trajectory in from index_split_by_add_force_flag
    flipped_add_force = np.concatenate(flipped_add_force)   # Concatenate together to make its shape as (-1, )
    if timelag != '0':
      flipped_add_force[-1] = 0 # If adding timelag, the last trajectory which is a lag flag is up should be ignored for the end_of_trajectory flag since it's not the real end point of trajectory
    if args.shift_eot:
      flipped_add_force = flipped_add_force.astype(int)
      flipped_add_force = np.concatenate((np.zeros(1), flipped_add_force))
      flipped_add_force[-2] = 1
      trajectory_split[i]['end_of_trajectory'] = flipped_add_force[:-1] # + unflip_add_force.astype(int)    # Assign to new columns
    else:
      trajectory_split[i]['end_of_trajectory'] = flipped_add_force.astype(int) # + unflip_add_force.astype(int)    # Assign to new columns
    # print(unflip_add_force + flipped_add_force)
    # print(flipped_add_force)
  return trajectory_split

=======
  return trajectory_split

def get_end_of_trajectory_flag(trajectory_split, timelag):
  for i in range(len(trajectory_split)):
    unflip_add_force = trajectory_split[i]['add_force_flag'].values # Get the unflip add_force_flag value columns on the trajectory i-th index
    index_split_by_add_force_flag = list(trajectory_split[i].loc[trajectory_split[i]['add_force_flag'] == True].index)[:] # remove the first trajectory and the last trajectory
    index_split_by_add_force_flag.append(len(unflip_add_force) + index_split_by_add_force_flag[0]) # Index of each trajectory
    index_split_by_add_force_flag = np.array(index_split_by_add_force_flag) - index_split_by_add_force_flag[0]  # Re-index every row to start from 0
    flipped_add_force = [np.flip(unflip_add_force[index_split_by_add_force_flag[j]:index_split_by_add_force_flag[j+1]]) for j in range(len(index_split_by_add_force_flag)-1)]   # Get each trajectory in from index_split_by_add_force_flag
    flipped_add_force = np.concatenate(flipped_add_force)   # Concatenate together to make its shape as (-1, )
    if timelag != '0':
      flipped_add_force[-1] = 0 # If adding timelag, the last trajectory which is a lag flag is up should be ignored for the end_of_trajectory flag since it's not the real end point of trajectory
    trajectory_split[i]['end_of_trajectory'] = flipped_add_force.astype(int) # + unflip_add_force.astype(int)    # Assign to new columns
    # print(unflip_add_force + flipped_add_force)
    # print(flipped_add_force)
  return trajectory_split

>>>>>>> 089ae3e542e8ea85ecac4da4d9d04183a5910226
def generate_constant_num_continuous_trajectory(trajectory_df, index_split_by_flag, num_continuous_trajectory, traj_type, timelag):
  # For the trajectory into continuous trajectory
  threshold_lengths = 12 # Remove some trajectory that cause from applying multiple force at a time (Threshold of applying force is not satisfied)
  temp_trajectory = []
  # For the timelag may access the trajectory at index_split_by_flag[end_index+1] and can cause the out-of-range error so need to -2 to reserve the last trajectory for timelag
  for i in range(0, int(len(index_split_by_flag)/(num_continuous_trajectory))-2):
    # [i] value will loop to get sequence (0, 1), (1, 2), (2, 3), (3, 4) ... to multiply with num_continuous_trajectory to get the index of ending trajectory
    start_index = i * num_continuous_trajectory   # Index to point out where to start in index_split_by_flag
    end_index = (i+1) * num_continuous_trajectory # Index to point out where to stop in index_split_by_flag
    # Adding timelag
    timelag_offset = get_timelag_offset(trajectory_df=trajectory_df, traj_type=traj_type, timelag=timelag, end_index=end_index, index_split_by_flag=index_split_by_flag)
    # Check the lengths of every trajectory before forming the continuous need to longer then threshold_lengths
    thresholding_lengths = [len(trajectory_df[traj_type].iloc[index_split_by_flag[start_index + j]:index_split_by_flag[start_index+j+1]]) for j in range(num_continuous_trajectory)]
    if all(length_traj > threshold_lengths for length_traj in thresholding_lengths):  # All length pass the condition
      temp_trajectory.append(trajectory_df[traj_type].iloc[index_split_by_flag[start_index]:index_split_by_flag[end_index]+timelag_offset, :])   # Append to the list(Will be list of dataframe)
  return temp_trajectory

def generate_random_num_continuous_trajectory(trajectory_df, index_split_by_flag, num_continuous_trajectory, traj_type, timelag):
  # For the trajectory into continuous trajectory
  threshold_lengths = 12 # Remove some trajectory that cause from applying multiple force at a time (Threshold of applying force is not satisfied)
  temp_trajectory = []
  random_continuous_length = np.arange(1, 8)
  total_trajectory = len(index_split_by_flag) - 2   # For the timelag may access the trajectory at index_split_by_flag[end_index+1] and can cause the out-of-range error so need to -2 to reserve the last trajectory for timelag
  ptr_index_split = 0 # Pointer to the trajectory
  while total_trajectory > 0:
    # Random the continuous trajectory length
    num_continuous_trajectory = np.random.choice(random_continuous_length)
    if total_trajectory - num_continuous_trajectory <= 0:
      # No trajectory left...
      break
    # [i] value will loop to get sequence (0, 1), (1, 2), (2, 3), (3, 4) ... to multiply with num_continuous_trajectory to get the index of ending trajectory
    start_index = ptr_index_split
    end_index = ptr_index_split + num_continuous_trajectory # Index to point out where to stop in index_split_by_flag
    timelag_offset = get_timelag_offset(trajectory_df=trajectory_df, traj_type=traj_type, timelag=timelag, end_index=end_index, index_split_by_flag=index_split_by_flag)
    # Check the lengths of every trajectory before forming the continuous need to longer then threshold_lengths
    thresholding_lengths = [len(trajectory_df[traj_type].iloc[index_split_by_flag[start_index + j]:index_split_by_flag[start_index+j+1]]) for j in range(num_continuous_trajectory)]
    if all(length_traj > threshold_lengths for length_traj in thresholding_lengths):  # All length pass the condition
      temp_trajectory.append(trajectory_df[traj_type].iloc[index_split_by_flag[start_index]:index_split_by_flag[end_index]+timelag_offset, :])   # Append to the list(Will be list of dataframe)
    # Move the pointer
    ptr_index_split += num_continuous_trajectory
    # Update the length of total_trajectory
    total_trajectory -= num_continuous_trajectory

  return temp_trajectory

def get_timelag_offset(trajectory_df, traj_type, index_split_by_flag, end_index, timelag):
  if int(len(trajectory_df[traj_type].iloc[index_split_by_flag[end_index]:index_split_by_flag[end_index+1]])/2) == 0:
    return 0
  else:
    if timelag == 'half':
      timelag_offset = int(len(trajectory_df[traj_type].iloc[index_split_by_flag[end_index]:index_split_by_flag[end_index+1]])/2)
    elif timelag == 'quater':
      timelag_offset = int(len(trajectory_df[traj_type].iloc[index_split_by_flag[end_index]:index_split_by_flag[end_index+1]])/4)
    elif timelag == 'oneeight':
      timelag_offset = np.ceil(len(trajectory_df[traj_type].iloc[index_split_by_flag[end_index]:index_split_by_flag[end_index+1]])/8).astype(int)
    elif timelag == 'random':
      timelag_offset = np.random.choice(range(int(len(trajectory_df[traj_type].iloc[index_split_by_flag[end_index]:index_split_by_flag[end_index+1]])/2)))
    else :
      timelag_offset = int(timelag)
  return timelag_offset

def addGravityColumns(trajectory_npy):
  stacked_gravity = [np.concatenate((trajectory_npy[i], np.array([-9.81]*len(trajectory_npy[i])).reshape(-1, 1)), axis=1) for i in range(trajectory_npy.shape[0])]
  stacked_gravity = np.array(stacked_gravity)
  return stacked_gravity

def get_col_names(dataset_folder, i):
  with open(dataset_folder + '/configFile_camParams_Trial{}.json'.format(i)) as json_file:
    loaded_config = json.load(json_file)
    col_names = loaded_config["col_names"]
    projectionMatrix = np.array(loaded_config["mainCameraParams"]["projectionMatrix"]).reshape(4, 4)
    worldToCameraMatrix = np.array(loaded_config["mainCameraParams"]["worldToCameraMatrix"]).reshape(4, 4)
    width = loaded_config["mainCameraParams"]["width"]
    height = loaded_config["mainCameraParams"]["height"]

    return col_names, {'projectionMatrix':projectionMatrix, 'worldToCameraMatrix':worldToCameraMatrix, 'width':width, 'height':height}

def get_savepath(output_path, dataset_folder):
  if output_path == None:
    output_path = dataset_folder
  else:
    output_path = args.output_path
    if not os.path.exists(output_path):
      os.makedirs(output_path)
  return output_path

if __name__ == '__main__':
  # Argument for preprocessing
  parser = argparse.ArgumentParser(description="Ball trajectory-preprocessing script")
  parser.add_argument('--dataset_path', type=str, help='Specify path to dataset', required=True)
  parser.add_argument('--split_by', type=str, help='Specify the flag for split', default='add_force_flag')
  parser.add_argument('--output_path', type=str, help='Specify output path to save dataset')
  parser.add_argument('--no_zero_ground', dest='force_zero_ground_flag', help='Input the flag that make all rolling trajectory stay on the ground(Force y=0)', action='store_false')
  parser.add_argument('--zero_ground', dest='force_zero_ground_flag', help='Input the flag that make all rolling trajectory stay on the ground(Force y=0)', action='store_true')
  parser.add_argument('--num_continuous_trajectory', type=int, help='Keep the continuous of trajectory', default=1)
  parser.add_argument('--random_num_continuous', dest='random_sampling_mode', help='Generate the random number of continuous trajectory', action='store_true')
  parser.add_argument('--constant_num_continuous', dest='random_sampling_mode', help='Generate the constant number of continuous trajectory', action='store_false')
  parser.add_argument('--timelag', dest='timelag', help='Timelag for input some part of next trajectory', default=0)
  parser.add_argument('--process_trial_index', dest='process_trial_index', help='Process trial at given idx only', default=None)
  parser.add_argument('--noise', dest='noise', help='Noise flag for adding noise and project to get noised pixel coordinates', action='store_true')
  parser.add_argument('--no_noise', dest='noise', help='Noise flag for adding noise and project to get noised pixel coordinates', action='store_false')
  parser.add_argument('--vis_noise', dest='vis_noise', help='Visualize effect of Noise', action='store_true')
  parser.add_argument('--no_vis_noise', dest='vis_noise', help='Visualize effect of Noise', action='store_false')
  parser.add_argument('--masking', dest='masking', help='Masking of Noise', action='store_true')
  parser.add_argument('--no_masking', dest='masking', help='Masking of Noise', action='store_false')
<<<<<<< HEAD
  parser.add_argument('--selected_features', dest='selected_features', help='Specify the selected features columns(eot, og, ', nargs='+', required=True)
  parser.add_argument('--selected_cams', dest='selected_cams', help='Specify the selected cams(main, along, top)', type=str, required=True)
  parser.add_argument('--selected_space', dest='selected_space', help='Specify the selected spaces(ndc, screen)', type=str, required=True)
  parser.add_argument('--shift_eot', dest='shift_eot', help='Shift eot flag by 1 position', default=False, action='store_true')
=======
  parser.add_argument('--og_flag', dest='on_ground_flag', help='Data has the on ground flag', action='store_true')
  parser.add_argument('--no_og_flag', dest='on_ground_flag', help='Data has the on ground flag', action='store_false')
>>>>>>> 089ae3e542e8ea85ecac4da4d9d04183a5910226

  args = parser.parse_args()
  # List trial in directory
  dataset_folder = sorted(glob.glob(args.dataset_path + "/*/"))
  pattern = r'(Trial_[0-9]+)+'
  print("Dataset : ", [re.findall(pattern, dataset_folder[i]) for i in range(len(dataset_folder))])
  if args.process_trial_index is not None:
    # Use only interesting trial : Can be edit the trial index in process_trial_index.txt
    with open(args.process_trial_index) as f:
      # Split the text input of interested trial index into list of trial index
      trial_index = f.readlines()[-1].split()
      # Create the pattern for regex following this : (10)|(11)|(12) ===> match the any trial from 10, 11 or 12
      pattern_trial_index= ['({}\/)'.format(trial_index[i]) for i in range(len(trial_index))]
      # Add it into full pattern of regex : r'(Trial_((10)|(11)|(12))+)+/' ===> Need to add '/' alphabet to prevent the regex match Trial_1=Trial_10 instead of only Trial_1
      pattern_trial_index = r'(Trial_({})+)+'.format('|'.join(pattern_trial_index))
      # filter the dataset folder which is not in the trial_index
      filter_trial_index = [re.search(pattern_trial_index, dataset_folder[i]) for i in range(len(dataset_folder))]
      # for i in range(len(filter_trial_index)):
        # print(filter_trial_index[i])

      dataset_folder = [dataset_folder[i] for i in range(len(filter_trial_index)) if filter_trial_index[i] is not None]
  else :
    # Use all trial
    trial_index = [re.findall(r'[0-9]+', re.findall(pattern, dataset_folder[i])[0])[0] for i in range(len(dataset_folder))]
  print("Trial index : ", trial_index)
  if args.random_sampling_mode:
    print("Mode : Random number of continuous trajectory with timelag = {}, force_zero_ground_flag = {} and noise flag = {}".format(args.timelag, args.force_zero_ground_flag, args.noise))
  else:
    print("Mode : Constant number of continuous trajectory with n = {}, timelag = {}, force_zero_ground_flag = {} and noise flag = {}".format(args.num_continuous_trajectory, args.timelag, args.force_zero_ground_flag, args.noise))

  trajectory_type = ["Rolling", "Projectile", "MagnusProjectile", "Mixed"]
  print(dataset_folder)
  for i in tqdm.tqdm(range(len(dataset_folder)), desc="Loading dataset"):
    output_path = get_savepath(args.output_path, dataset_folder[i])
    # Read json for column names
    col_names, camera_config = get_col_names(dataset_folder[i], trial_index[i])
    trajectory_df = {}
    for traj_type in trajectory_type:
      if os.path.isfile(dataset_folder[i] + "/{}Trajectory_Trial{}.csv".format(traj_type, trial_index[i])):
        trajectory_df[traj_type] = pd.read_csv(dataset_folder[i] + "/{}Trajectory_Trial{}.csv".format(traj_type, trial_index[i]), names=col_names, skiprows=1, delimiter=',')
    print("Trajectory type in Trial{} : {}".format(trial_index[i], trajectory_df.keys()))

    # Split the trajectory by flag
    trajectory_split = split_by_flag(trajectory_df=trajectory_df, trajectory_type=trajectory_df.keys(), flag="add_force_flag", force_zero_ground_flag=args.force_zero_ground_flag, num_continuous_trajectory=args.num_continuous_trajectory, random_sampling_mode=args.random_sampling_mode, timelag=args.timelag)

    if args.noise:
      # Add gaussian noise to the trajectory
      trajectory_split = add_noise(trajectory_split=trajectory_split, trajectory_type=trajectory_df.keys(), camera_config=camera_config)
    # Cast to npy format
    trajectory_npy = computeDisplacement(trajectory_split=trajectory_split, trajectory_type=trajectory_df.keys())
    # Save to npy format
    for traj_type in trajectory_df.keys():
      # Adding Gravity columns
      trajectory_npy[traj_type] = addGravityColumns(trajectory_npy[traj_type])
      # Write each trajectory
      np.save(file=output_path + "/{}Trajectory_Trial{}.npy".format(traj_type, trial_index[i]), arr=trajectory_npy[traj_type])

