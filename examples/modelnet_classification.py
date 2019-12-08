# coding=utf-8
import numpy as np
from sklearn.metrics import accuracy_score

from thumoon.generation import gen_knn_hg
from thumoon.learning import multi_hg_trans_infer
from thumoon.utils import print_log

from examples.data_helper import load_modelnet


def main():
    print_log("loading data")
    X_train, X_test, y_train, y_test = load_modelnet(selected_mod=(0, 1))

    X = [np.vstack((X_train[imod], X_test[imod])) for imod in range(len(X_train))]
    y = np.concatenate((y_train, -1 * np.ones_like(y_test)))

    print_log("generating hypergraph")
    hg_list = [
        gen_knn_hg(X[imod], n_neighbors=10)
        for imod in range(len(X_train))
    ]

    print_log("learning on hypergraph")
    y_predict = multi_hg_trans_infer(hg_list, y, lbd=100)

    print_log("accuracy: {}".format(accuracy_score(y_test, y_predict)))


if __name__ == "__main__":
    main()

