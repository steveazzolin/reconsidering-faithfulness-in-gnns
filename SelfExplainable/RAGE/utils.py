import numpy as np
from torchmetrics.functional import accuracy as torch_accuracy
from torchmetrics.functional import auroc as torch_auc
from torchmetrics.functional import average_precision as torch_ap
from torchmetrics.functional import r2_score as torch_r2_score
from torchmetrics.functional import mean_absolute_error as torch_mae
from torchmetrics.functional import mean_squared_error as torch_mse

from sklearn import preprocessing


def prepare_ground_pred(ground, pred):
    ground = ground.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()  # n x #classes

    assert np.isclose(len(pred), pred.sum())  # are predictions probabilities

    num_classes = pred.shape[1]
    ground = preprocessing.label_binarize(ground, classes=list(range(num_classes)))
    if num_classes == 2:  # binary classification
        ground = np.hstack((1 - ground, ground))

    return ground, pred


def prepare_ground_pred_regression(ground, pred):
    ground = ground.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()  # n x #classes
    return ground, pred


def auc(ground, pred,task="multiclass"):
    return torch_auc(pred, ground, num_classes=pred.shape[1],task=task).item()


def ap(ground, pred,task="multiclass"):
    return torch_ap(pred, ground, num_classes=pred.shape[1],task=task).item()


def accuracy(ground, pred,task="multiclass"):
    return torch_accuracy(pred, ground, num_classes=pred.shape[1],task=task).item()


def r_squared(ground, pred):
    return torch_r2_score(pred, ground).item()


def mse(ground, pred):
    return torch_mse(pred, ground).item()


def mae(ground, pred):
    return torch_mae(pred, ground).item()
