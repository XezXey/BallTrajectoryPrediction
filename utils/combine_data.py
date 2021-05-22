import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', dest='data_path', required=True)
parser.add_argument('--file_idx', dest='file_idx', nargs='+', required=True)
parser.add_argument('--sample', dest='sample', type=int, required=True)
parser.add_argument('--mode', dest='mode', type=str, required=True)

args = parser.parse_args()

if __name__ == '__main__':
  data = []
  for i in range(len(args.file_idx)):
    data_file = glob.glob('{}/*{}.npy'.format(args.data_path, args.file_idx[i]))
    if len(data_file) > 0:
      data.append(np.load(data_file[0], allow_pickle=True))
    else:
      print("[#] Trial {} not found".format(args.file_idx[i]))

  for i in range(len(data)):
    print("[#] Shape : ", data[i].shape)

  print("Limit : ", np.concatenate(data).shape)
  data = np.concatenate(data)
  limit = data.shape[0]
  start = 0
  i = 1
  while limit > args.sample:
    data_ = data[start:start+args.sample]
    start += args.sample
    limit -= args.sample

    print("Fold {} : {}".format(i, data_.shape))
    np.save(file='{}/MixedTrajectory_{}_fold{}.npy'.format(args.data_path, args.mode, i), arr=data_)
    i+= 1
