from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
seq_len = int(input("Seq Length : "))
ramp = np.linspace(0, 1, seq_len)
# print(t)
# ramp = 0.5 * (signal.sawtooth(2 * np.pi * seq_len * t) + 1)
plt.plot(np.arange(0, seq_len), ramp, label='Before thresholding')

threshold = 0.4
idx = np.where(ramp < threshold)[0]
ramp[idx] = threshold
plt.plot(np.arange(0, seq_len), ramp, label='After thresholding')
plt.show()
