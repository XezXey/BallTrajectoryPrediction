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
  return trajectory_npy

def split_by_flag(trajectory_df, trajectory_type, flag='add_force_flag', force_zero_ground_flag=False):
  threshold_lengths = 5 # Remove some trajectory that cause from applying multiple force at a time (Threshold of applying force is not satisfied)
  trajectory_split = trajectory_df
  for traj_type in trajectory_type:
    if traj_type=='Rolling' and force_zero_ground_flag is True:
      trajectory_df[traj_type].iloc[:, 1] = trajectory_df[traj_type].iloc[:, 1] * 0.0
    trajectory_df[traj_type] = trajectory_df[traj_type].replace({"True":True, "False":False})
    # Split each dataframe by using the flag == True as an index of starting point
    index_split_by_flag = list(trajectory_df[traj_type].loc[trajectory_df[traj_type][flag] == True].index)
    # Store splitted dataframe in list (Not use the first and last trajectory : First one can be bug if the ball is not on the 100% ground, Last one is the the complete trajectory)
    trajectory_split[traj_type] = [trajectory_df[traj_type].iloc[index_split_by_flag[i]:index_split_by_flag[i+1], :] for i in range(1, len(index_split_by_flag)-1) if len(trajectory_df[traj_type].iloc[index_split_by_flag[i]:index_split_by_flag[i+1], :]) > threshold_lengths]
    # print("Each trajectory length : ", [trajectory_split[traj_type][i].shape for i in range(len(trajectory_split[traj_type]))])
  return trajectory_split

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
  parser.add_argument('--force_zero_ground_flag', type=bool, help='Input the flag the make all rolling trajectory stay on the ground(Force y=0)', default=False)
  args = parser.parse_args()
  # List trial in directory
  dataset_folder = sorted(glob.glob(args.dataset_path + "/*/"))
  pattern = r'(Trial_[0-9])+'
  print(re.findall(pattern, dataset_folder[0]))
  trial_index = [re.findall(r'[0-9]+', re.findall(pattern, dataset_folder[i])[0])[0] for i in range(len(dataset_folder))]
  print(trial_index)
  trajectory_type = ["Rolling", "Projectile", "MagnusProjectile"]
  for i in tqdm.tqdm(range(len(dataset_folder)), desc="Loading dataset"):
    output_path = get_savepath(args.output_path, dataset_folder[i])
    # Read json for column names
    col_names = get_col_names(dataset_folder[i], trial_index[i])
    trajectory_df = {"Rolling" : pd.read_csv(dataset_folder[i] + "/RollingTrajectory_Trial{}.csv".format(trial_index[i]), names=col_names, skiprows=1, delimiter=','),
                      "Projectile" : pd.read_csv(dataset_folder[i] + "/ProjectileTrajectory_Trial{}.csv".format(trial_index[i]), names=col_names, skiprows=1, delimiter=','),
                      "MagnusProjectile" : pd.read_csv(dataset_folder[i] + "/MagnusProjectileTrajectory_Trial{}.csv".format(trial_index[i]), names=col_names, skiprows=1, delimiter=',')}
    # Split the trajectory by flag
    trajectory_split = split_by_flag(trajectory_df, trajectory_type, flag="add_force_flag", force_zero_ground_flag=args.force_zero_ground_flag)
    # Cast to npy format
    trajectory_npy = computeDisplacement(trajectory_split, trajectory_type)
    # Save to npy format
    for traj_type in trajectory_type:
      np.save(file=output_path + "/{}Trajectory_Trial{}.npy".format(traj_type, trial_index[i]), arr=trajectory_npy[traj_type])
    # for key, values in trajectory_npy.items():
      # print("{} trajectory : #N = {} trajectory".format(key, values.shape))

