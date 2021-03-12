import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import tqdm
import seaborn as sns
parser = argparse.ArgumentParser(description='Analyze the data')
parser.add_argument('--data_folder', dest='data_folder', type=str, help='Path to data file', required=True)
parser.add_argument('--feature', dest='feature', type=str, help='Feature to analyze', nargs='+', required=True)
parser.add_argument('--cumsum', dest='cumsum', help='Perform a cummulative summation', action='store_true', default=False)
parser.add_argument('--cmp_folder', dest='cmp_folder', help='Compare to', type=str, default=None)
args = parser.parse_args()

def get_feature_columns():
  feature_col = []
  feature_name = []
  if 'x' in args.feature:
    feature_col.append(0)
    feature_name.append('x')
  if 'y' in args.feature:
    feature_col.append(1)
    feature_name.append('y')
  if 'z' in args.feature:
    feature_col.append(2)
    feature_name.append('z')
  if 'u' in args.feature:
    feature_col.append(3)
    feature_name.append('u')
  if 'v' in args.feature:
    feature_col.append(4)
    feature_name.append('v')
  if 'depth' in args.feature:
    feature_col.append(5)
    feature_name.append('depth')

  return feature_col, feature_name

def compare_to(feature_col, feature_name):
  data_folder = sorted(glob.glob(args.cmp_folder + '/*.npy'))
  for each_file in tqdm.tqdm(data_folder, desc="[Comparing...] Loading dataset"):
    feature_cmp = []
    data = np.load(each_file, allow_pickle=True)
    for trajectory in data:
      if args.cumsum:
        feature_cmp.append(np.cumsum(trajectory[:, feature_col]))
      else:
        feature_cmp.append(trajectory[1:, feature_col])
    feature_cmp = np.concatenate(feature_cmp)
    sns.kdeplot(feature_cmp, label=each_file.split('/')[-1], clip=[-10, 10], fill=True)
    # sns.histplot(feature_cmp, label=each_file.split('/')[-1])


if __name__ == '__main__':
  data_folder = sorted(glob.glob(args.data_folder + '/*.npy'))
  feature_col, feature_name = get_feature_columns()
  for each_file in tqdm.tqdm(data_folder, desc="Loading dataset"):
    print("="*100)
    data = np.load(each_file, allow_pickle=True)
    for idx in range(len(feature_col)):
      feature = []
      for trajectory in data:
        if args.cumsum:
          feature.append(np.cumsum(trajectory[:, feature_col[idx]]))
        else:
          feature.append(trajectory[1:, feature_col[idx]])
      feature = np.concatenate(feature)
      plt.plot(feature)
      plt.show()
      # Visualize
      if args.cmp_folder is not None:
        compare_to(feature_col=feature_col[idx], feature_name=feature_name[idx])
      sns.kdeplot(feature, label=each_file.split('/')[-1], clip=[-10, 10], fill=True)
      # sns.histplot(feature, label=each_file.split('/')[-1])
      plt.xlabel("Displacement of {}".format(feature_name[idx]))
      # plt.xlim(-10, 10)
      plt.legend()
      plt.show()
      print("Analyzing {} : {}".format(feature_name[idx], each_file))
      print("===>Mean : ", np.mean(feature))
      print("===>SD : ", np.std(feature))
      print("===>Max : ", np.max(feature))
      print("===>Min : ", np.min(feature))
    print("="*100)
