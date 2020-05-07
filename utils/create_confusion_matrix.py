from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# [TN, FN], [FP, TP]
# Bounce
class_names = ['EOT = 0', 'EOT = 1']
confusion_matrix_bounce = np.array([[1179069, 7],
                                    [2, 5669]])

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix_bounce,
                                class_names=class_names)
plt.title('Bounce Trajectory : Mixed')
plt.xlabel('Predicted')
plt.ylabel('GroundTruth')
plt.show()

# NoBounce
confusion_matrix_nobounce = np.array([[525349, 4],
                                      [2, 4581]])

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix_nobounce,
                                class_names=class_names)
plt.title('NoBounce Trajectory : Mixed')
plt.xlabel('Predicted')
plt.ylabel('GroundTruth')
plt.show()
