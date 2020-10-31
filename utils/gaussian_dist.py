import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Plot between -10 and 10 with .001 steps.
# x_axis = np.arange(-10, 10, 0.001)
h = 1
mean = 0
sigma = 1
# x = np.linspace(-3 * sigma + mean, 3 * sigma + mean, 100000)
# x = np.random.normal(loc=mean, scale=sigma, size=100)
while True:
  seq_len = int(input("Sequence length (Auto Inclusive) : "))
  if seq_len % 2 != 0:
    seq_len += 1
  step = 10/seq_len
  print("Step : ", step)
  x = np.arange(-5, 5+step, step)
  # print(x)
  gaussian = h * np.exp(-((x-mean)**2)/(2*(sigma**2)))
  mask = np.invert(np.isclose(gaussian, np.finfo(float).eps))
  # print(mask)
  gaussian *= mask
  # print(len(gaussian))
  # print(max(gaussian))
  # print(min(gaussian))
  mid_idx = np.where(gaussian==1.0)[0][0]
  print("Length : ", len(gaussian[:mid_idx]), len([mid_idx]), len(gaussian[mid_idx+1:]))
  plt.plot(gaussian)
  plt.plot(mask)
  plt.axvline(x=np.where(gaussian==1.0)[0], c='r')
  plt.show()

