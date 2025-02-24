import os
import torch

import numpy as np

from math import sqrt
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error


def eval_rocauc(y_true, y_pred=None):

    if isinstance(y_true, dict):
        input_dict = y_true
        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)

    rocauc_list = []

    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            y = np.array(y_true[:, i], dtype=np.float64)
            y[y < 0] = np.nan
            is_labeled = y == y
            rocauc_list.append(
                roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            )

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return {'rocauc': sum(rocauc_list) / len(rocauc_list)}


def eval_rmse(y_true, y_pred=None):
    if isinstance(y_true, dict):
        input_dict = y_true
        y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)

    return {'rmse': sqrt(mean_squared_error(y_true, y_pred))}

