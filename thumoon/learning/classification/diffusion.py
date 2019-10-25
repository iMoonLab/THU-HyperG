# coding=utf-8
import numpy as np
import scipy.sparse as sparse

from thumoon.base import HyperG
from thumoon.utils import print_log, init_label_matrix


def cross_diffusion_infer(hg_list, y, iter, log=True):
    """ cross diffusion from the "Cross Diffusion on Multi-hypergraph
    for Multi-modal 3D Object Recognition" paper.
    :param hg_list: list, list of HyperG instance
    :param y: numpy array, shape = (n_nodes,)
    :param iter: int, iteration times of diffusion
    :param log: bool
    :return:
    """
    assert isinstance(hg_list, list)

    n_hg = len(hg_list)
    assert n_hg >= 2

    Y = init_label_matrix(y)

    P = [None for _ in range(n_hg)]
    F1 = [np.copy(Y) for _ in range(n_hg)]
    F2 = [np.copy(Y) for _ in range(n_hg)]

    # calculate transition matrix P
    for hg_idx in range(n_hg):
        H = hg_list[hg_idx].incident_matrix()
        w = hg_list[hg_idx].hyperedge_weights()
        INVDE = hg_list[hg_idx].inv_edge_degrees()

        S = H.dot(sparse.diags(w)).dot(INVDE).dot(H.T)
        INVS_S = sparse.diags(1 / S.sum(axis=1).A.reshape(-1))
        P[hg_idx] = INVS_S.dot(S)

    for i_iter in range(iter):
        for hg_idx in range(n_hg):
            F2[hg_idx] = P[hg_idx].dot(F1[(hg_idx + 1) % n_hg])
            F2[hg_idx][y != -1] = Y[y != -1]
        for hg_idx in range(n_hg):
            F1[hg_idx] = F2[hg_idx]

        if log:
            print_log("Iter: {}".format(i_iter))

    F = np.zeros_like(Y)
    for hg_idx in range(n_hg):
        F += F1[hg_idx]
    F = F / n_hg
    predict_y = np.argmax(F, axis=1).reshape(-1)

    return predict_y[y == -1]
