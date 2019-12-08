# coding=utf-8
import numpy as np
from sklearn.metrics import accuracy_score

from thumoon.generation import gen_knn_hg
from thumoon.learning import trans_infer
from thumoon.utils import print_log

from examples.data_helper import load_MSRGesture3D


def main():
    print_log("loading data")
    X_train, X_test, y_train, y_test = load_MSRGesture3D()

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, -1 * np.ones_like(y_test)])

    print_log("generating hypergraph")
    hg = gen_knn_hg(X, n_neighbors=4)

    print_log("learning on hypergraph")
    y_predict = trans_infer(hg, y, lbd=100)
    print_log("accuracy: {}".format(accuracy_score(y_test, y_predict)))


if __name__ == "__main__":
    main()







