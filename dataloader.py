import torch as pt
import numpy as np
import argparse
from tqdm import tqdm
import glob
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):

  def __init__(self, data_path, trajectory_type):
    # Initialization
    self.data_path = {"Rolling" : glob.glob(data_path + "/Rolling*.npy"),
                      "MagnusProjectile" : glob.glob(data_path + "/MagnusProjectile*.npy"),
                      "Projectile" : glob.glob(data_path + "/Projectile*.npy")}
    self.trajectory_type = trajectory_type
    # print(self.data_path["Rolling"])
    # print(self.data_path["MagnusProjectile"])
    # print(self.data_path["Projectile"])
    # Load data
    self.trajectory_dataset = {"Rolling" : [np.load(self.data_path["Rolling"][i], allow_pickle=True) for i in tqdm(range(len(self.data_path["Rolling"])), desc="Rolling")],
                               "Projectile" : [np.load(self.data_path["Projectile"][i], allow_pickle=True) for i in tqdm(range(len(self.data_path["Projectile"])), desc="Projectile")],
                               "MagnusProjectile" : [np.load(self.data_path["MagnusProjectile"][i], allow_pickle=True) for i in tqdm(range(len(self.data_path["MagnusProjectile"])), desc="MagnusProjectile")]}

    # Select trajectory type
    print("===============================Dataset shape===============================")
    for trajectory_type in self.trajectory_dataset.keys():
      self.trajectory_dataset[trajectory_type] = np.concatenate([self.trajectory_dataset[trajectory_type][i] for i in range(len(self.trajectory_dataset[trajectory_type]))])
      print("{} : {}".format(trajectory_type, self.trajectory_dataset[trajectory_type].shape))
    print("===========================================================================")

  def __len__(self):
    # Denotes the total number of samples
    return len(self.trajectory_dataset[self.trajectory_type])

  def __getitem__(self, idx):
    # Generates one batch of dataset by trajectory
    # print("At idx={} : {}".format(idx, self.trajectory_dataset[self.trajectory_type][idx].shape))
    # print(type(self.trajectory_dataset[self.trajectory_type]))
    return self.trajectory_dataset[self.trajectory_type][idx]

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## Get sequence lengths
    # print("Batch size : ", len(batch))
    lengths = pt.tensor([trajectory.shape[0] for trajectory in batch])
    ## Padding
    # print("Lengths of sequence : ", lengths)
    batch = [pt.Tensor(trajectory) for trajectory in batch]
    batch = pt.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    # print("Batch shape : ", batch.shape)
    ## compute mask
    mask = (batch != 0)
    # print("Mask shape : ", mask.shape)
    return batch, lengths, mask

if __name__ == '__main__':
  print("************************************TESTING DATALOADER CLASS************************************")
  # For test the module
  parser = argparse.ArgumentParser(description='Trajectory dataloader')
  parser.add_argument('--dataset_path', dest='dataset_path', type=str, help='Specify path to dataset')
  parser.add_argument('--batch_size', type=int, help='Specify batch size', default=50)
  parser.add_argument('--trajectory_type', type=str, help="Specify trajectory type(Projectile, Rolling, MagnusProjectile)", default='Projectile')
  args = parser.parse_args()
  trajectory_dataset = TrajectoryDataset(args.dataset_path, trajectory_type=args.trajectory_type)
  trajectory_dataloader = DataLoader(trajectory_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn_padd, pin_memory=True)
  print("===============================Summary Batch (batch_size = {})===============================".format(args.batch_size))
  for key, values in enumerate(trajectory_dataloader):
    batch, lengths, mask = values
    print("Batch {} : batch={}, lengths={}, mask={}".format(key, batch.shape, lengths.shape, mask.shape))

    # Test RNN/LSTM Step
    # 1.Pack the padded
    packed = pt.nn.utils.rnn.pack_padded_sequence(batch, batch_first=True, lengths=lengths, enforce_sorted=False)
    # 2.RNN/LSTM model
    # 3.Unpack the packed
    unpacked = pt.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
    print("Unpacked equality : ", pt.eq(batch, unpacked[0]).all())
  print("===========================================================================")
