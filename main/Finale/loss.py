import torch as pt
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly
import wandb

# GPU initialization
if pt.cuda.is_available():
  device = pt.device('cuda')
else:
  device = pt.device('cpu')

def GravityLoss(pred, gt, mask, lengths):
  # Compute the 2nd finite difference of the y-axis to get the gravity should be equal in every time step
  gravity_constraint_penalize = pt.tensor([0.])
  count = 0
  # Gaussian blur kernel for not get rid of the input information
  gaussian_blur = pt.tensor([0.25, 0.5, 0.25], dtype=pt.float32).view(1, 1, -1).to(device)
  # Kernel weight for performing a finite difference
  kernel_weight = pt.tensor([-1., 0., 1.], dtype=pt.float32).view(1, 1, -1).to(device)
  # Apply Gaussian blur and finite difference to gt
  for i in range(gt.shape[0]):
    # print(gt[i][:lengths[i]+1, 1])
    # print(gt[i][:lengths[i]+1, 1].shape)
    if gt[i][:lengths[i], 1].shape[0] < 6:
      print("The trajectory is too shorter to perform a convolution")
      continue
    gt_yaxis_1st_gaussian_blur = pt.nn.functional.conv1d(gt[i][:lengths[i], 1].view(1, 1, -1), gaussian_blur)
    gt_yaxis_1st_finite_difference = pt.nn.functional.conv1d(gt_yaxis_1st_gaussian_blur, kernel_weight)
    gt_yaxis_2nd_gaussian_blur = pt.nn.functional.conv1d(gt_yaxis_1st_finite_difference, gaussian_blur)
    gt_yaxis_2nd_finite_difference = pt.nn.functional.conv1d(gt_yaxis_2nd_gaussian_blur, kernel_weight)
    # Apply Gaussian blur and finite difference to gt
    pred_yaxis_1st_gaussian_blur = pt.nn.functional.conv1d(pred[i][:lengths[i], 1].view(1, 1, -1), gaussian_blur)
    pred_yaxis_1st_finite_difference = pt.nn.functional.conv1d(pred_yaxis_1st_gaussian_blur, kernel_weight)
    pred_yaxis_2nd_gaussian_blur = pt.nn.functional.conv1d(pred_yaxis_1st_finite_difference, gaussian_blur)
    pred_yaxis_2nd_finite_difference = pt.nn.functional.conv1d(pred_yaxis_2nd_gaussian_blur, kernel_weight)
    # Compute the penalize term
    # print(gt_yaxis_2nd_finite_difference.shape, pred_yaxis_2nd_finite_difference.shape)
    if gravity_constraint_penalize.shape[0] == 1:
      gravity_constraint_penalize = ((gt_yaxis_2nd_finite_difference - pred_yaxis_2nd_finite_difference)**2).reshape(-1, 1)
    else:
      gravity_constraint_penalize = pt.cat((gravity_constraint_penalize, ((gt_yaxis_2nd_finite_difference - pred_yaxis_2nd_finite_difference)**2).reshape(-1, 1)))

  return pt.mean(gravity_constraint_penalize)

def BelowGroundPenalize(pred, gt, mask, lengths):
  # Penalize when the y-axis is below on the ground
  pred = pred * mask
  below_ground_mask = pred[..., 1] < -0.5
  below_ground_constraint_penalize = pt.mean((pred[..., 1] * below_ground_mask)**2)
  return below_ground_constraint_penalize

def TrajectoryLoss(pred, gt, mask, lengths=None, delmask=True):
  # L2 loss of reconstructed trajectory
  x_trajectory_loss = (pt.sum((((gt[..., 0] - pred[..., 0]))**2) * mask[..., 0]) / pt.sum(mask[..., 0]))
  y_trajectory_loss = (pt.sum((((gt[..., 1] - pred[..., 1]))**2) * mask[..., 1]) / pt.sum(mask[..., 1]))
  z_trajectory_loss = (pt.sum((((gt[..., 2] - pred[..., 2]))**2) * mask[..., 2]) / pt.sum(mask[..., 2]))
  return x_trajectory_loss + y_trajectory_loss + z_trajectory_loss


def DepthLoss(pred, gt, mask, lengths):
  depth_loss = (pt.sum((((gt - pred))**2) * mask) / pt.sum(mask))
  return depth_loss

def EndOfTrajectoryLoss(pred, gt, startpos, mask, lengths, flag='Train'):
  # Here we use output mask so we need to append the startpos to the pred before multiplied with mask(already included the startpos)
  pred *= mask
  gt *= mask

  # Log the precision, recall, confusion_matrix and using wandb
  gt_log = gt.clone().cpu().detach().numpy()
  pred_log = pred.clone().cpu().detach().numpy()
  eot_metrics_log(gt=gt_log, pred=pred_log, lengths=lengths.cpu().detach().numpy(), flag=flag)

  # Implement from scratch
  # Flatten and concat all trajectory together
  gt = pt.cat(([gt[i][:lengths[i]+1] for i in range(startpos.shape[0])]))
  pred = pt.cat(([pred[i][:lengths[i]+1] for i in range(startpos.shape[0])]))
  # Class weight for imbalance class problem
  pos_weight = pt.sum(gt == 0)/pt.sum(gt==1)
  neg_weight = 1
  # Prevent of pt.log(-value)
  eps = 1e-10
  # Calculate the BCE loss
  eot_loss = pt.mean(-((pos_weight * gt * pt.log(pred + eps)) + (neg_weight * (1-gt)*pt.log(1-pred + eps))))
  return eot_loss

def eot_metrics_log(gt, pred, lengths, flag):
  pred = pred > 0.5
  # Output of confusion_matrix.ravel() = [TN, FP ,FN, TP]
  cm_each_trajectory = np.array([confusion_matrix(y_pred=pred[i][:lengths[i], :], y_true=gt[i][:lengths[i]]).ravel() for i in range(lengths.shape[0])])
  n_accepted_trajectory = np.sum(np.logical_and(cm_each_trajectory[:, 1]==0., cm_each_trajectory[:, 2] == 0.))
  cm_batch = np.sum(cm_each_trajectory, axis=0)
  tn, fp, fn, tp = cm_batch
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1_score = 2 * (precision * recall) / (precision + recall)
  wandb.log({'{} Precision'.format(flag):precision, '{} Recall'.format(flag):recall, '{} F1-score'.format(flag):f1_score, '{}-#N accepted trajectory(Perfect EOT without FN, FP)'.format(flag):n_accepted_trajectory})