# coding=utf-8
import numpy as np
import scipy.sparse as sparse

from thumoon.base import HyperG, IMHL
from thumoon.utils import print_log, init_label_matrix, calculate_accuracy


def inductive_fit(hg, y, lbd, mu, eta, max_iter, log=True):
    """ inductive multi-hypergraph learning from the "Inductive
    Multi-Hypergraph Learning and Its Application on View-Based
    3D Object Classification"
    (you should call the inductive_fit first and then
    call the inductive_predict to predict unlabeled instances)
    :param hg: instance of HyperG or list
    :param y: numpy array, shape = (n_nodes,)
    :param lbd: float, the positive tradeoff parameter of empirical loss.
    :param mu: float, the positive tradeoff parameter of the regularizer on projection matrix.
    :param eta: float, the positive tradeoff parameter of the l2 norm on hypergraph weights
    :param max_iter: int, maximum iteration times of alternative optimization.
    :param log: bool
    :return: instance of IMHL
    """
    assert isinstance(hg, (HyperG, list))
    assert isinstance(y, (np.ndarray, list))

    if isinstance(hg, HyperG):
        hg_list = [hg]
    else:
        hg_list = hg

    n_hg = len(hg_list)

    Y = init_label_matrix(y)

    M = [None for _ in range(n_hg)]
    omega = np.zeros(n_hg)
    loss = np.zeros(n_hg)

    for hg_idx in range(n_hg):
        if log:
            print_log("processing I_HG :{}".format(hg_idx))

        X = hg_list[hg_idx].node_features()
        L = hg_list[hg_idx].laplacian()

        _, n_features = X.shape
        INVU = np.eye(n_features)

        for i_iter in range(max_iter):
            # fix U, update M
            A = sparse.csr_matrix.dot(X.T, L).dot(X) + lbd * X.T.dot(X)
            TMP = np.linalg.inv(A.dot(INVU) + mu * np.eye(n_features))
            M[hg_idx] = lbd * INVU.dot(TMP).dot(X.T).dot(Y)

            # fix M, update U
            invu = np.sqrt(np.sum(np.power(M[hg_idx], 2.), axis=1)).reshape(-1)
            INVU = 2 * np.diag(invu)

        g_reg_term = np.trace(M[hg_idx].T.dot(X.T).dot(L.dot(X).dot(M[hg_idx])))
        emp_loss_term = np.power(np.linalg.norm(X.dot(M[hg_idx]) - Y), 2)
        m_reg_term = np.sum([np.linalg.norm(M[hg_idx][i, :]) for i in range(n_features)])
        i_loss = g_reg_term + lbd * emp_loss_term + mu * m_reg_term

        if log:
            print_log("I_HG: {}; loss:{:.5f}".format(hg_idx, i_loss))

        loss[hg_idx] = i_loss

    for hg_idx in range(n_hg):
        omega[hg_idx] = 1./n_hg + np.sum(loss)/(2*n_hg*eta) - loss[hg_idx]/(2*eta)

    if log:
        print("hypergraph weights:{}".format(omega))

    return IMHL(M, omega)


def inductive_predict(X, model):
    """ inductive multi-hypergraph learning
    :param X: numpy array, shape = (n_test, n_features)
    :param model: instance of IMHL
    :return: numpy array, shape = (n_test, ), predicted labels of test instances
    """

    if isinstance(X, np.ndarray):
        X = [X]
    M = model.M
    omega = model.omega

    Y = np.zeros((X[0].shape[0], M[0].shape[1]))
    n_mod = len(X)

    for i_mod in range(n_mod):
        Y += omega[i_mod] * (X[i_mod].dot(M[i_mod]))

    predict_y = np.argmax(Y, axis=1).reshape(-1)

    return predict_y

