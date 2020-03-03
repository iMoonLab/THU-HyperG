# coding=utf-8
import time
from itertools import product
import numpy as np
from scipy.optimize import linear_sum_assignment



def print_log(message):
    """
    :param message: str,
    :return:
    """
    print("[{}] {}".format(time.strftime("%Y-%m-%d %X", time.localtime()), message))


def init_label_matrix(y):
    """
    :param y: numpy array, shape = (n_nodes,) -1 for the unlabeled data, 0,1,2.. for the labeled data
    :return:
    """
    y = y.reshape(-1)
    labels = list(np.unique(y))

    if -1 in labels:
        labels.remove(-1)

    n_nodes = y.shape[0]
    Y = np.ones((n_nodes, len(labels))) * (1/len(labels))
    for idx, label in enumerate(labels):
        Y[np.where(y == label), :] = 0
        Y[np.where(y == label), idx] = 1

    return Y


def calculate_accuracy(F, y_test):
    predict_y = np.argmax(F, axis=1).reshape(-1)
    return sum(predict_y == y_test) / len(predict_y)


def iou_socre(pred, target):
    """
    :param pred:
    :param target:
    :return:
    """
    ious = []
    n_class = target.max() + 1
    
    # IOU for background class ("0")
    for c in range(1, n_class):
        pred_idx = pred == c
        target_idx = target == c
        intersection = (pred_idx & target_idx).sum()
        union = (pred_idx | target_idx).sum()
        ious.append((intersection + 1e-6)/(union + 1e-6))

    return ious


def minmax_scale(array, ranges=(0., 1.)):
    """
    normalize to [min, max], default is [0., 1.]
    :param array: ndarray
    :param ranges: tuple, (min, max)
    :return:
    """
    _min = ranges[0]
    _max = ranges[1]
    return (_max - _min) * (array - array.min()) / (array.max() - array.min()) + _min


def gather_patch_ft(x, patch_size):
    """
    :param x: M x N x C
    :param patch_size: row x column
    :return:
    """
    assert len(x.shape) == 3
    assert len(patch_size) == 2

    x_row_num, x_col_num = x.shape[:2]
    x = x.reshape(-1, x.shape[2])
    x = np.concatenate([np.zeros(x.shape[1])[np.newaxis, :], x])

    # generate out index
    out_idx = []
    center_row, center_col = (patch_size[0] + 1) // 2 - 1, (patch_size[1] + 1) // 2 - 1
    x_idx = np.arange(x_row_num * x_col_num).reshape(x_row_num, x_col_num)

    x_idx_pad = np.zeros((x_row_num + patch_size[0] - 1, x_col_num + patch_size[1] - 1))
    x_idx_pad[center_row:center_row + x_row_num, center_col:center_col + x_col_num] = x_idx + 1

    for _row, _col in product(range(patch_size[0]), range(patch_size[1])):
        out_idx.append(x_idx_pad[_row:_row + x_row_num, _col:_col + x_col_num].reshape(-1, 1))
    out_idx = np.concatenate(out_idx, axis=1).astype(np.long)   # MN x kk

    # apply out index
    out = x[out_idx.reshape(-1)]    # MNkk x C
    out = out.reshape(x_row_num, x_col_num, -1) # M x N x kkC
    return out


def calculate_clustering_accuracy(y_gnd, y_pred):
    """
    :param y_gnd:
    :param y_pred:
    :return:
    """
    y_pred = y_pred.reshape(-1)
    y_gnd = y_gnd.reshape(-1)
    
    n_samples = y_gnd.shape[0]
    n_class = np.unique(y_gnd).shape[0]
    
    M = np.zeros((n_class, n_class))

    for i in range(n_samples):
        r = y_gnd[i]
        c = y_pred[i]
        M[r, c] += 1

    row_idx, col_idx = linear_sum_assignment(-M)

    map = np.zeros((n_class, n_class))
    map[row_idx, col_idx] = 1.

    acc = np.sum(M * map) / n_samples

    return acc



if __name__ == "__main__":
    pass




