from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# [TN, FN], [FP, TP]
# Bounce
class_names = ['EOT = 0', 'EOT = 1']
confusion_matrix_bounce = np.array([[1179062, 11],
                                    [9, 5665]])

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix_bounce,
                                class_names=class_names)
plt.title('Bounce Trajectory : Mixed')
plt.xlabel('Predicted')
plt.ylabel('GroundTruth')
plt.show()

# NoBounce
confusion_matrix_nobounce = np.array([[525350, 8],
                                      [1, 4577]])

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix_nobounce,
                                class_names=class_names)
plt.title('NoBounce Trajectory : Mixed')
plt.xlabel('Predicted')
plt.ylabel('GroundTruth')
plt.show()
