import numpy as np
import json
import argparse
import pandas as pd
import json
import tqdm
import glob
import os
import re

def computeDisplacement(trajectory_split, trajectory_type):
  # Compute the displacement
  drop_cols = ["add_force_flag", "outside_flag", "trajectory_type", "t"]
  trajectory_npy = trajectory_split.copy()
  for traj_type in trajectory_type:
    # Keep the first point as a starting point for performing a cumsum to retrieve whole trajectory 
    # Remove by get rid of the np.vstack
    trajectory_npy[traj_type] = [np.vstack((trajectory_split[traj_type][i].drop(drop_cols, axis=1).iloc[0, :].values,
                                           np.diff(trajectory_split[traj_type][i].drop(drop_cols, axis=1).loc[:, :].values, axis=0))) for i in range(len(trajectory_split[traj_type]))]
    # Cast to ndarray (Bunch of trajectory)
    trajectory_npy[traj_type] = np.array([trajectory_npy[traj_type][i] for i in range(len(trajectory_npy[traj_type]))])
    # Remove some dataset that goes below the ground (Error from unity)
    trajectory_npy[traj_type] = remove_below_ground_trajectory(trajectory=trajectory_npy[traj_type], traj_type=traj_type)
  return trajectory_npy

def remove_below_ground_trajectory(trajectory, traj_type):
  # Loop over the trajectory to remove any trajectory that goes below the ground
  # Also remove the trajectory that is outside the field and droping to the ground
  count=0
  remove_idx = []
  for idx in range(trajectory.shape[0]):
    traj_cumsum_temp = np.cumsum(trajectory[idx][:, 1], axis=0)
    if np.any(traj_cumsum_temp <= -1) == True:
      remove_idx.append(idx)
      count+=1
  print("{}===>Remove the below ground trajectory : {} at {}".format(traj_type, count, remove_idx))
  trajectory = np.delete(trajectory.copy(), obj=remove_idx)
  return trajectory

def split_by_flag(trajectory_df, trajectory_type, num_continuous_trajectory, flag='add_force_flag', force_zero_ground_flag=False, random_sampling_mode=False):
  trajectory_split = trajectory_df
  for traj_type in trajectory_type:
    if traj_type=='Rolling' and force_zero_ground_flag is True:
      trajectory_df[traj_type].iloc[:, 1] = trajectory_df[traj_type].iloc[:, 1] * 0.0
    trajectory_df[traj_type] = trajectory_df[traj_type].replace({"True":True, "False":False})
    # Split each dataframe by using the flag == True as an index of starting point
    index_split_by_flag = list(trajectory_df[traj_type].loc[trajectory_df[traj_type][flag] == True].index)[0:-1] # remove the first trajectory and the last trajectory
    # Store splitted dataframe in list (Not use the first and last trajectory : First one can be bug if the ball is not on the 100% ground, Last one is the the complete trajectory)
    # Split the trajectory that shorter than threhsold 12 points
    # trajectory_split[traj_type] = [trajectory_df[traj_type].iloc[index_split_by_flag[i]:index_split_by_flag[i+1], :] for i in range(1, len(index_split_by_flag)-1) if len(trajectory_df[traj_type].iloc[index_split_by_flag[i]:index_split_by_flag[i+1], :]) > threshold_lengths]
    if random_sampling_mode:
      trajectory_split[traj_type] = generate_random_num_continuous_trajectory(trajectory_df=trajectory_df, index_split_by_flag=index_split_by_flag, num_continuous_trajectory=num_continuous_trajectory, traj_type=traj_type)
    else:
      trajectory_split[traj_type] = generate_constant_num_continuous_trajectory(trajectory_df=trajectory_df, index_split_by_flag=index_split_by_flag, num_continuous_trajectory=num_continuous_trajectory, traj_type=traj_type)
  return trajectory_split

def generate_constant_num_continuous_trajectory(trajectory_df, index_split_by_flag, num_continuous_trajectory, traj_type):
  # For the trajectory into continuous trajectory
  threshold_lengths = 12 # Remove some trajectory that cause from applying multiple force at a time (Threshold of applying force is not satisfied)
  temp_trajectory = []
  for i in range(0, int(len(index_split_by_flag)/(num_continuous_trajectory))-1):
    # [i] value will loop to get sequence (0, 1), (1, 2), (2, 3), (3, 4) ... to multiply with num_continuous_trajectory to get the index of ending trajectory
    start_index = i * num_continuous_trajectory   # Index to point out where to start in index_split_by_flag
    end_index = (i+1) * num_continuous_trajectory # Index to point out where to stop in index_split_by_flag
    # Check the lengths of every trajectory before forming the continuous need to longer then threshold_lengths
    thresholding_lengths = [len(trajectory_df[traj_type].iloc[index_split_by_flag[start_index + j]:index_split_by_flag[start_index+j+1]]) for j in range(num_continuous_trajectory)]
    if all(length_traj > threshold_lengths for length_traj in thresholding_lengths):  # All length pass the condition
      temp_trajectory.append(trajectory_df[traj_type].iloc[index_split_by_flag[start_index]:index_split_by_flag[end_index], :])   # Append to the list(Will be list of dataframe)
  return temp_trajectory

def generate_random_num_continuous_trajectory(trajectory_df, index_split_by_flag, num_continuous_trajectory, traj_type):
  # For the trajectory into continuous trajectory
  threshold_lengths = 12 # Remove some trajectory that cause from applying multiple force at a time (Threshold of applying force is not satisfied)
  temp_trajectory = []
  random_continuous_length = np.arange(1, 5)
  total_trajectory = len(index_split_by_flag) - 1
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
    # Check the lengths of every trajectory before forming the continuous need to longer then threshold_lengths
    thresholding_lengths = [len(trajectory_df[traj_type].iloc[index_split_by_flag[start_index + j]:index_split_by_flag[start_index+j+1]]) for j in range(num_continuous_trajectory)]
    if all(length_traj > threshold_lengths for length_traj in thresholding_lengths):  # All length pass the condition
      temp_trajectory.append(trajectory_df[traj_type].iloc[index_split_by_flag[start_index]:index_split_by_flag[end_index], :])   # Append to the list(Will be list of dataframe)
    # Move the pointer
    ptr_index_split += num_continuous_trajectory
    # Update the length of total_trajectory
    total_trajectory -= num_continuous_trajectory

  return temp_trajectory


def addGravityColumns(trajectory_npy):
  # print(trajectory_npy.shape)
  stacked_gravity = [np.concatenate((trajectory_npy[i], np.array([-9.81]*len(trajectory_npy[i])).reshape(-1, 1)), axis=1) for i in range(trajectory_npy.shape[0])]
  stacked_gravity = np.array(stacked_gravity)
  return stacked_gravity

def get_col_names(dataset_folder, i):
  with open(dataset_folder + '/configFile_camParams_Trial{}.json'.format(i)) as json_file:
    col_names = json.load(json_file)["col_names"]
    return col_names

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
  parser.add_argument('--force_zero_ground_flag', type=bool, help='Input the flag that make all rolling trajectory stay on the ground(Force y=0)', default=False)
  parser.add_argument('--num_continuous_trajectory', type=int, help='Keep the continuous of trajectory', default=1)
  parser.add_argument('--random_num_continuous', dest='random_sampling_mode', help='Generate the random number of continuous trajectory', action='store_true')
  parser.add_argument('--constant_num_continuous', dest='random_sampling_mode', help='Generate the constant number of continuous trajectory', action='store_false')
  args = parser.parse_args()
  # List trial in directory
  dataset_folder = sorted(glob.glob(args.dataset_path + "/*/"))
  pattern = r'(Trial_[0-9]+)+'
  print(re.findall(pattern, dataset_folder[0]))
  trial_index = [re.findall(r'[0-9]+', re.findall(pattern, dataset_folder[i])[0])[0] for i in range(len(dataset_folder))]
  print(trial_index)
  if args.random_sampling_mode:
    print("Mode : Random number of continuous trajectory")
  else:
    print("Mode : Constant number of continuous trajectory")
  trajectory_type = ["Rolling", "Projectile", "MagnusProjectile"]
  for i in tqdm.tqdm(range(len(dataset_folder)), desc="Loading dataset"):
    output_path = get_savepath(args.output_path, dataset_folder[i])
    # Read json for column names
    col_names = get_col_names(dataset_folder[i], trial_index[i])
    trajectory_df = {"Rolling" : pd.read_csv(dataset_folder[i] + "/RollingTrajectory_Trial{}.csv".format(trial_index[i]), names=col_names, skiprows=1, delimiter=','),
                      "Projectile" : pd.read_csv(dataset_folder[i] + "/ProjectileTrajectory_Trial{}.csv".format(trial_index[i]), names=col_names, skiprows=1, delimiter=','),
                      "MagnusProjectile" : pd.read_csv(dataset_folder[i] + "/MagnusProjectileTrajectory_Trial{}.csv".format(trial_index[i]), names=col_names, skiprows=1, delimiter=',')}
    # Split the trajectory by flag
    trajectory_split = split_by_flag(trajectory_df, trajectory_type, flag="add_force_flag", force_zero_ground_flag=args.force_zero_ground_flag, num_continuous_trajectory=args.num_continuous_trajectory, random_sampling_mode=args.random_sampling_mode)
    # Cast to npy format
    trajectory_npy = computeDisplacement(trajectory_split, trajectory_type)
    # Save to npy format
    for traj_type in trajectory_type:
      # Adding Gravity columns
      trajectory_npy[traj_type] = addGravityColumns(trajectory_npy[traj_type])
      # Write each trajectory
      np.save(file=output_path + "/{}Trajectory_Trial{}.npy".format(traj_type, trial_index[i]), arr=trajectory_npy[traj_type])

