import torch as pt
import numpy as np
import math
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)

for i in np.arange(0.0, 361.0, 1.):
  degree = pt.tensor(i)
  print('='*100)
  print("[Pytorch-Rad] Degree : {}, Sin : {}, Asin : {}".format(degree, pt.sin(degree * math.pi/180.0), pt.asin(pt.sin(degree * math.pi/180.0)) * 180.0/math.pi))
  print("[Numpy-Rad] Degree : {}, Sin : {}, Asin : {}".format(degree, np.sin(degree * math.pi/180.0), np.arcsin(np.sin(degree * math.pi/180.0)) * 180.0/math.pi))

  print("[Pytorch-Rad] Degree : {}, Cos : {}, Acos : {}".format(degree, pt.cos(degree * math.pi/180.0), pt.acos(pt.cos(degree * math.pi/180.0)) * 180.0/math.pi))
  print("[Numpy-Rad] Degree : {}, Cos : {}, Acos : {}".format(degree, np.cos(degree * math.pi/180.0), np.arccos(np.cos(degree * math.pi/180.0)) * 180.0/math.pi))

  print("[Pytorch-Deg] Degree : {}, Sin : {}, Asin : {}".format(degree, pt.sin(degree), pt.asin(pt.sin(degree * math.pi/180.0)) * 180.0/math.pi))
  print("[Numpy-Deg] Degree : {}, Sin : {}, Asin : {}".format(degree, np.sin(degree), np.arcsin(np.sin(degree * math.pi/180.0)) * 180.0/math.pi))

  print("[Pytorch-Rad] Degree : {}, Cos : {}, Acos : {}".format(degree, pt.cos(degree), pt.acos(pt.cos(degree * math.pi/180.0)) * 180.0/math.pi))
  print("[Numpy-Rad] Degree : {}, Cos : {}, Acos : {}".format(degree, np.cos(degree), np.arccos(np.cos(degree * math.pi/180.0)) * 180.0/math.pi))
