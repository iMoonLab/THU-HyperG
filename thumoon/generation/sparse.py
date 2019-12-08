# coding=utf-8

import numpy as np
from sklearn.metrics import pairwise_distances
import scipy.sparse as sparse
import cvxpy as cp
from cvxpy.error import SolverError

from thumoon.base import HyperG
from thumoon.utils import print_log

# TODO: 1. elastic net hypergraph


def gen_l1_hg(X, gamma, n_neighbors, log=False, with_feature=False):
    """
    :param X: numpy array, shape = (n_samples, n_features)
    :param gamma: float, the tradeoff parameter of the l1 norm on representation coefficients
    :param n_neighbors: int,
    :param log: bool
    :param with_feature: bool, optional(default=False)
    :return: instance of HyperG
    """

    assert n_neighbors >= 1.
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2

    n_nodes = X.shape[0]
    n_edges = n_nodes

    m_dist = pairwise_distances(X)
    m_neighbors = np.argsort(m_dist)[:, 0:n_neighbors+1]

    edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors+1)).reshape(-1)
    node_idx = []
    values = []

    for i_edge in range(n_edges):
        if log:
            print_log("processing edge {} ".format(i_edge))

        neighbors = m_neighbors[i_edge].tolist()
        if i_edge in neighbors:
            neighbors.remove(i_edge)
        else:
            neighbors = neighbors[:-1]

        P = X[neighbors, :]
        v = X[i_edge, :]

        # cvxpy
        x = cp.Variable(P.shape[0], nonneg=True)
        objective = cp.Minimize(cp.norm((P.T@x).T-v, 2) + gamma * cp.norm(x, 1))
        # objective = cp.Minimize(cp.norm(x@P-v, 2) + gamma * cp.norm(x, 1))
        prob = cp.Problem(objective)
        try:
            prob.solve()
        except SolverError:
            prob.solve(solver='SCS', verbose=False)

        node_idx.extend([i_edge] + neighbors)
        values.extend([1.] + x.value.tolist())

    node_idx = np.array(node_idx)
    values = np.array(values)

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))

    if with_feature:
        return HyperG(H, X=X)

    return HyperG(H)
