import os
import pickle as pkl

import numpy as np
import scipy.io as scio
import SimpleITK as sitk
from sklearn.preprocessing import normalize

from thumoon.utils import minmax_scale
from thumoon.utils import print_log

DATA_DIR = os.path.join(os.path.dirname(__file__), 'datasets')


def load_myocardium(test_idx=[4]):
    heart_seg_dir = os.path.join(DATA_DIR, 'myocardiumSeg')
    ori = os.listdir(os.path.join(heart_seg_dir, 'ori'))

    X = []
    y = []

    for name in ori:
        ori_img = sitk.ReadImage(os.path.join(heart_seg_dir, "ori/{}".format(name)))
        ori_ary = minmax_scale(sitk.GetArrayFromImage(ori_img).squeeze()) # (y, x)
        X.append(ori_ary)

        seg_img = sitk.ReadImage(os.path.join(heart_seg_dir, "seg/{}".format(name)))
        seg_ary = sitk.GetArrayFromImage(seg_img).squeeze()
        y.append(seg_ary)

    X = np.stack(X)
    y = np.stack(y)

    training_idx = [i for i in range(X.shape[0]) if i not in test_idx]

    X_train = X[training_idx]
    X_test = X[test_idx]
    y_train = y[training_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def load_modelnet(selected_mod):
    print_log("selected mod:{}".format(str(selected_mod)))
    modelnet40_dir = os.path.join(DATA_DIR, "modelnet40")
    X_train = pkl.load(open(os.path.join(modelnet40_dir, 'modelnet_train_fts.pkl'), 'rb'))
    X_test = pkl.load(open(os.path.join(modelnet40_dir, 'modelnet_test_fts.pkl'), 'rb'))

    y_train = pkl.load(open(os.path.join(modelnet40_dir, 'modelnet_train_lbls.pkl'), 'rb'))
    y_test = pkl.load(open(os.path.join(modelnet40_dir, 'modelnet_test_lbls.pkl'), 'rb'))

    X_train = [X_train[imod] for imod in selected_mod]
    X_test = [X_test[imod] for imod in selected_mod]

    if len(selected_mod) == 1:
        X_train = X_train[0]
        X_test = X_test[0]

    return X_train, X_test, np.array(y_train), np.array(y_test)


def load_MSRGesture3D(i_train=2, i_test = 0):
    msr_gesture_dir = os.path.join(DATA_DIR, "MSRGesture3D")
    data = scio.loadmat(os.path.join(msr_gesture_dir, 'MSRGesture3D.mat'))
    all_indices = scio.loadmat(os.path.join(msr_gesture_dir, 'MSRGesture3DTrainIndex.mat'))['trainIndex']

    i_indices = all_indices[i_test, i_train].reshape(-1)
    X = data['X']
    X = normalize(X)
    y = np.array(data['Y'], dtype=np.int).reshape(-1)
    y = y - np.min(y)

    X_train = X[i_indices == 1]
    X_test = X[i_indices == 0]
    y_train = y[i_indices == 1]
    y_test = y[i_indices == 0]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pass
