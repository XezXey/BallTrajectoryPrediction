import torch
import numpy as np
import argparse
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):

  def __init__(self, data_path):
    # Initialization
    self.data_path = {"Rolling" : glob.glob(data_path + "/Rolling*.npy"),
                      "MagnusProjectile" : glob.glob(data_path + "/MagnusProjectile*.npy"),
                      "Projectile" : glob.glob(data_path + "/Projectile*.npy")}
    # print(self.data_path["Rolling"])
    # print(self.data_path["MagnusProjectile"])
    # print(self.data_path["Projectile"])
    # Load data
    self.trajectory_dataset = {"Rolling" : [np.load(self.data_path["Rolling"][i], allow_pickle=True) for i in tqdm(range(len(self.data_path["Rolling"])), desc="Rolling")],
                               "Projectile" : [np.load(self.data_path["Projectile"][i], allow_pickle=True) for i in tqdm(range(len(self.data_path["Projectile"])), desc="Projectile")],
                               "MagnusProjectile" : [np.load(self.data_path["MagnusProjectile"][i], allow_pickle=True) for i in tqdm(range(len(self.data_path["MagnusProjectile"])), desc="MagnusProjectile")]}

    # Select trajectory type
    for trajectory_type in self.trajectory_dataset.keys():
      self.trajectory_dataset[trajectory_type] = np.concatenate([self.trajectory_dataset[trajectory_type][i] for i in range(len(self.trajectory_dataset[trajectory_type]))])
      print("{} : {}".format(trajectory_type, self.trajectory_dataset[trajectory_type].shape))

  def __len__(self):
    # Denotes the total number of samples
    return len(self.trajectory_dataset["Projectile"])

  def __getitem__(self, idx):
    # Generates one batch of dataset by trajectory
    print(self.trajectory_dataset["Projectile"][:idx].shape)
    return self.trajectory_dataset["Projectile"][idx]


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Trajectory dataloader')
  parser.add_argument('--dataset_path', dest='dataset_path', type=str, help='Specify path to dataset')
  parser.add_argument('--batch_size', type=int, help='Specify batch size')
  args = parser.parse_args()
  traj_dataset = TrajectoryDataset(args.dataset_path)
  trajectory_data = DataLoader(traj_dataset, batch_size=4)
  for key, values in enumerate(trajectory_data):
    print(key, values.size())

