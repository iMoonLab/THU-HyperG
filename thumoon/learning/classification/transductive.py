# coding=utf-8
import time

import numpy as np
import cvxpy as cp
import scipy.sparse as sparse
from numpy.linalg import inv
from itertools import combinations
from sklearn.metrics import pairwise_distances

from thumoon.base import HyperG
from thumoon.utils import print_log, init_label_matrix, calculate_accuracy


def trans_infer(hg, y, lbd):
    """transductive inference from the "Learning with Hypergraphs:
     Clustering, Classification, and Embedding" paper

    :param hg: instance of HyperG
    :param y: numpy array, shape = (n_nodes,)
    :param lbd: float, the positive tradeoff parameter of empirical loss
    :return: numpy array, shape = (n_test, ), predicted labels of test instances
    """
    assert isinstance(hg, HyperG)

    Y = init_label_matrix(y)
    n_nodes = Y.shape[0]
    THETA = hg.theta_matrix()

    L2 = sparse.eye(n_nodes) - (1 / (1 + lbd)) * THETA
    F = ((lbd + 1) / lbd) * inv(L2.toarray()).dot(Y)

    predict_y = F.argmax(axis=1).reshape(-1)[y == -1]
    return predict_y


def hyedge_weighting_trans_infer(hg, y, lbd, mu, max_iter=15, log=True):
    """ hyperedge weighting from the "Visual-Textual Joint Relevance
    Learning for Tag-Based Social Image Search" paper

    :param hg: instance of HyperG
    :param y: numpy array, shape = (n_nodes,)
    :param lbd: float, the positive tradeoff parameter of empirical loss
    :param mu: float, the positive tradeoff parameter of hyperedge weights
    :param max_iter: int, maximum iteration times of alternative optimization
    :param log: bool
    :return: numpy array, shape = (n_test, ), predicted labels of test instances
    """
    assert max_iter > 0

    loss = []
    prev_F = None
    Y = init_label_matrix(y)

    for i_iter in range(max_iter):

        # update F
        if i_iter > 0:
            prev_F = F
        F = _transductive_one_step(hg, Y, lbd)

        # update w
        n_edges = hg.num_edges()
        H = hg.incident_matrix()
        DV = hg.node_degrees()
        DV2 = hg.inv_square_node_degrees()
        invde = hg.inv_edge_degrees().data.reshape(-1)
        dv = DV.data.reshape(-1)

        Tau = DV2.dot(H).toarray()
        C = np.zeros((n_edges, 1))
        for i in range(n_edges):
            C[i, 0] = -invde[i] * np.trace(
                np.sum(np.power(F.T.dot(Tau[:, i].reshape(-1, 1)), 2.), axis=0, keepdims=True)
            )
        # update w --- optimization
        Dense_H = H.toarray()
        w = cp.Variable(n_edges, nonneg=True)
        objective = cp.Minimize((1 / 2) * cp.quad_form(w, 2 * mu * np.eye(n_edges)) + C.T @ w)
        constraints = [Dense_H @ w == dv, w >= np.zeros(n_edges)]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        hg.update_hyedge_weights(w.value)

        # loss
        hg_reg_loss = np.trace(F.T.dot(hg.laplacian().dot(F)))
        emp_loss = np.linalg.norm(F - Y)
        w_loss = np.sum(np.power(w.value, 2.))
        i_loss = hg_reg_loss + lbd * emp_loss + w_loss
        loss.append(i_loss)

        # log
        if log:
            print_log("Iter: {}; loss:{:.5f}".format(i_iter, i_loss))

        if i_iter > 0 and (i_loss - loss[-2]) > 0.:
            print_log("Iteration stops after {} round.".format(i_iter))
            F = prev_F
            break

    predict_y = F.argmax(axis=1).reshape(-1)[y == -1]
    return predict_y


def dyna_hg_trans_infer(hg, y, lbd, stepsize, beta, max_iter, hsl_iter, log=True):
    """dynamic hypergraph structure learning from the "Dynamic Hypergraph
    Structure Learning" paper.
    :param hg: instance of HyperG
    :param y: numpy array, shape = (n_nodes,)
    :param lbd: float, the positive tradeoff parameter of empirical loss
    :param stepsize: float
    :param beta: float, the positive tradeoff parameter of regularizer on H in the feature space loss
    :param max_iter: int, maximum iteration times of alternative optimization.
    :param hsl_iter: int, the number of iterations in the process of updating hypergraph incident matrix
    :param log: bool
    :return: numpy array, shape = (n_test, ), predicted labels of test instances
    """
    if log:
        print_log("alpha:{}\tbeta:{}".format(stepsize, beta))

    n_nodes = hg.num_nodes()

    Y = init_label_matrix(y)

    a = np.ones(n_nodes)
    a[y != -1] = 1e-4
    A = sparse.diags(a)
    X = hg.node_features()

    assert X is not None, "hg instance should be constructed with the parameter of 'with_feature=True'"

    F = _dhsl_update_f(hg, Y, A, lbd)

    for i_iter in range(max_iter):
        # update H
        C = (1 - beta) * F.dot(F.T) + beta * X.dot(X.T)

        for i_hsl in range(hsl_iter):
            # sparse
            DV2 = hg.inv_square_node_degrees()
            INVDE = hg.inv_edge_degrees()
            W = sparse.diags(hg.hyperedge_weights())

            # dense
            H = hg.incident_matrix()
            WDHD = W.dot(INVDE).dot(H.T).dot(DV2).todense()
            H = H.todense()
            DCD = sparse.dia_matrix.dot(DV2.dot(C), DV2)

            term3 = 2 * sparse.coo_matrix.dot(DCD.dot(H), W.dot(INVDE))
            tmp = np.diag(H.T.dot(DCD).dot(H)).reshape(1, -1)
            term2 = - np.tile(sparse.csr_matrix.dot(tmp, W.dot(INVDE).dot(INVDE)), (n_nodes, 1))
            term1 = - DV2.dot(DV2).dot(DV2).dot(np.diag(H.dot(WDHD).dot(C)).reshape(-1, 1)).dot(
                hg.hyperedge_weights().reshape(1, -1)
            )

            dfH = term1 + term2 + term3
            H = H + stepsize * dfH
            H[H < 0] = 0
            H[H > 1] = 1
            H = sparse.coo_matrix(H)
            hg.update_incident_matrix(H)

        # update F
        F = _dhsl_update_f(hg, Y, A, lbd)

        if log:
            print_log("Iter: {}".format(i_iter))

    predict_y = F.argmax(axis=1).reshape(-1)[y == -1]
    return predict_y


def multi_hg_trans_infer(hg_list, y, lbd):
    """ multi-hypergraph transtductive infer
    :param hg_list: list, list of HyperG instance
    :param y: numpy array, shape = (n_nodes,)
    :param lbd: float, the positive tradeoff parameter of empirical loss
    :return: numpy array, shape = (n_test, ), predicted labels of test instances
    """
    assert isinstance(hg_list, list)
    Y = init_label_matrix(y)
    n_nodes = Y.shape[0]

    THETA = np.zeros((n_nodes, n_nodes))
    for i in range(len(hg_list)):
        THETA += hg_list[i].theta_matrix()

    L2 = sparse.eye(n_nodes) - (1 / (1 + lbd)) * THETA
    F = (((lbd + 1) / lbd) * inv(L2.A)).dot(Y)

    predict_y = F.argmax(axis=1).reshape(-1)[y == -1]
    return predict_y


def multi_hg_weighting_trans_infer(hg_list, y, lbd, mu, max_iter, log=True):
    """ multi-hypergraph weighting from the "3-D Object Retrieval and
    Recognition With Hypergraph Analysis"
    :param hg_list: list, list of HyperG instance
    :param y: numpy array, shape = (n_nodes,)
    :param lbd: float, the positive tradeoff parameter of empirical loss
    :param mu: float, the positive tradeoff parameter of the l2 norm on hypergraph weights
    :param max_iter: int, maximum iteration times of alternative optimization.
    :param log: bool
    :return: numpy array, shape = (n_test, ), predicted labels of test instances
    """
    assert max_iter > 0

    loss = []
    prev_F = None
    Y = init_label_matrix(y)
    n_hg = len(hg_list)
    n_nodes = len(y)
    omega = np.zeros(n_hg)
    hg_weights = np.ones(n_hg)/n_hg

    for i_iter in range(max_iter):
        print_log("hypergraph weights: {}".format(str(hg_weights)))

        # update F
        THETA = None
        for idx in range(len(hg_list)):
            if THETA is None:
                THETA = hg_weights[idx] * hg_list[idx].theta_matrix()
            else:
                THETA += hg_weights[idx] * hg_list[idx].theta_matrix()

        L2 = sparse.eye(n_nodes) - (1 / (1 + lbd)) * THETA
        F = ((lbd + 1) / lbd) * inv(L2.toarray()).dot(Y)

        # update hg_weight
        for idx in range(n_hg):
            omega[idx] = np.trace(F.T.dot(np.eye(n_nodes) + hg_list[idx].theta_matrix()).dot(F)) # TODO
        for idx in range(n_hg):
            hg_weights[idx] = 1. / n_hg + np.sum(omega) / (2 * mu * n_hg) - omega[idx] / (2 * mu)

        # loss
        i_loss = np.sum(omega * hg_weights) + lbd * np.linalg.norm(F - Y)
        loss.append(i_loss)

        if log:
            print_log("Iter: {}; loss:{:.5f}".format(i_iter, i_loss))

        if i_iter > 0 and (i_loss - loss[-2]) > 0.:
            print_log("Iteration stops after {} round.".format(i_iter))
            F = prev_F
            break
        prev_F = F

    predict_y = F.argmax(axis=1).reshape(-1)[y == -1]
    return predict_y


def tensor_hg_trans_infer(X, y, lbd, alpha, gamma, stepsize, max_iter=50, hsl_iter=10, log=True, stop=True):
    """ tensor-based (dynamic) hypergraph learning
    :param X: numpy array, shape = (n_nodes, n_features)
    :param y: numpy array, shape = (n_nodes,) -1 for the unlabeled data, 0,1,2.. for the labeled data
    :param lbd: float, the positive tradeoff parameter of empirical loss
    :param alpha: float,
    :param gamma: float,
    :param stepsize: float
    :param max_iter: int, maximum iteration times of alternative optimization
    :param hsl_iter: int, the number of iterations in the process of updating hypergraph tensor
    :param log: bool
    :param stop: boolean,
    :return: numpy array, shape = (n_test, ), predicted labels of test instances
    """
    if log:
        print_log("parameters: lambda:{}\talpha:{}\tgamma:{}\tstepsize:{}".format(lbd, alpha, gamma, stepsize))

    n_nodes = X.shape[0]

    # init Y
    Y = init_label_matrix(y)

    ceil_logn = np.ceil(np.log(n_nodes)).astype('int')

    # init tensor hypergraph
    T = np.ones((1, n_nodes * ceil_logn))

    mED = pairwise_distances(X)

    neighbors = np.asarray(np.argsort(mED, axis=1, kind='stable'))
    neighbors = neighbors[:, 1:(ceil_logn + 1)]

    # calculate S
    S = np.zeros((n_nodes, n_nodes))
    delta_omega = np.arange(2, ceil_logn + 2)
    delta_omega = np.tile(delta_omega, (n_nodes, 1))
    delta_omega = delta_omega.reshape((1, -1), order='F')

    t_iter = 0
    for i in range(ceil_logn):
        for j in range(n_nodes):
            if T[0, t_iter] == 0:
                continue

            clique = [j] + neighbors[j, 0:i + 1].tolist()
            indexs = np.array(list(combinations(clique, 2)))
            S[indexs[:, 0], indexs[:, 1]] = S[indexs[:, 0], indexs[:, 1]] + T[0, t_iter] / delta_omega[0, t_iter]

            t_iter = t_iter + 1

    S = S + S.T
    c = 1 / (1 + alpha)

    F = np.linalg.inv(np.eye(n_nodes) + (2 * c / lbd) * (np.diag(np.sum(S, axis=0)) - S)) @ Y

    T0 = T
    fH_value = []

    loss = []
    prev_F = None

    for i_iter in range(max_iter):
        FED = pairwise_distances(F)
        f = np.zeros((1, T.shape[1]))
        f_iter = 0
        for i in range(ceil_logn):
            for j in range(n_nodes):
                clique = [j] + neighbors[j, 0:i + 1].tolist()
                indexs = np.array(list(combinations(clique, 2)))
                tmp1 = np.sum(FED[indexs[:, 0], indexs[:, 1]], axis=0)
                tmp2 = np.sum(mED[indexs[:, 0], indexs[:, 1]], axis=0)
                f[0, f_iter] = (tmp1 + alpha * tmp2) / ((1 + alpha) * delta_omega[0, f_iter])
                f_iter = f_iter + 1

        for iter2 in range(hsl_iter):
            dfT = f + 2 * gamma * (T - T0)
            T = T - stepsize * dfT
            T[T < 0] = 0
            T[T > 1] = 1

        fH_value.append(
            (f @ T.T).reshape(-1)[0] + lbd * np.power(np.linalg.norm(F - Y, ord='fro'), 2.) + gamma * np.power(
                np.linalg.norm(T - T0, ord='fro'), 2))

        S = np.zeros((n_nodes, n_nodes))
        t_iter = 0
        for i in range(ceil_logn):
            for j in range(n_nodes):
                if T[0, t_iter] == 0:
                    continue

                clique = [j] + neighbors[j, 0:i + 1].tolist()
                indexs = np.array(list(combinations(clique, 2)))
                S[indexs[:, 0], indexs[:, 1]] = S[indexs[:, 0], indexs[:, 1]] + T[0, t_iter] / delta_omega[0, t_iter]

                t_iter = t_iter + 1
        S = S + S.T
        c = 1 / (1 + alpha)
        F = np.linalg.inv(np.eye(n_nodes) + (2 * c / lbd) * (np.diag(np.sum(S, 0)) - S)) @ Y

        fh = (f @ T.T).reshape(-1)[0] + lbd * np.power(np.linalg.norm(F - Y, ord='fro'), 2.) + gamma * np.power(
            np.linalg.norm(T - T0, ord='fro'), 2)
        if log:
            loss.append(fh)
            print_log("Iter: {}; loss:{:.5f}".format(i_iter, fh))

        fH_value.append(fh)

        if stop:
            if len(loss) >= 2 and loss[-1] > loss[-2]:
                print_log("Stop at iteration :{}".format(i_iter))
                F = prev_F
                break
            prev_F = F

    predict_y = np.argmax(F, axis=1).reshape(-1)[y == -1]
    return predict_y


def _transductive_one_step(hg, Y, lbd):
    n_nodes = hg.num_nodes()

    THETA = hg.theta_matrix()
    L2 = sparse.eye(n_nodes) - (1 / (1 + lbd)) * THETA
    F = ((lbd + 1) / lbd) * inv(L2.toarray()).dot(Y)

    return F


def _dhsl_update_f(hg, Y, A, lbd):
    n_nodes = hg.num_nodes()

    THETA = hg.theta_matrix()
    L = sparse.eye(n_nodes) - THETA
    F = inv((A + 1 / lbd * L).toarray()).dot(A.dot(Y))

    return F