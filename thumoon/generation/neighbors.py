# coding=utf-8
import numpy as np
from sklearn.metrics import pairwise_distances
import scipy.sparse as sparse

from thumoon.base import HyperG
from thumoon.utils import print_log


def gen_knn_hg(X, n_neighbors, is_prob=True, with_feature=False):
    """
    :param X: numpy array, shape = (n_samples, n_features)
    :param n_neighbors: int,
    :param is_prob: bool, optional(default=True)
    :param with_feature: bool, optional(default=False)
    :return: instance of HyperG
    """

    assert isinstance(X, (np.ndarray, list))
    assert n_neighbors > 0

    X = np.array(X)
    n_nodes = X.shape[0]
    n_edges = n_nodes

    m_dist = pairwise_distances(X)

    # top n_neighbors+1
    m_neighbors = np.argpartition(m_dist, kth=n_neighbors+1, axis=1)
    m_neighbors_val = np.take_along_axis(m_dist, m_neighbors, axis=1)

    m_neighbors = m_neighbors[:, :n_neighbors+1]
    m_neighbors_val = m_neighbors_val[:, :n_neighbors+1]

    # check
    for i in range(n_nodes):
        if not np.any(m_neighbors[i, :] == i):
            m_neighbors[i, -1] = i
            m_neighbors_val[i, -1] = 0.

    node_idx = m_neighbors.reshape(-1)
    edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors+1)).reshape(-1)

    if not is_prob:
        values = np.ones(node_idx.shape[0])
    else:
        avg_dist = np.mean(m_dist)
        m_neighbors_val = m_neighbors_val.reshape(-1)
        values = np.exp(-np.power(m_neighbors_val, 2.) / np.power(avg_dist, 2.))

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))

    w = np.ones(n_edges)

    if with_feature:
        return HyperG(H, w=w, X=X)

    return HyperG(H, w=w)


def gen_epsilon_ball_hg(X, ratio, is_prob=True, with_feature=False):
    """
    :param X: numpy array, shape = (n_samples, n_features)
    :param ratio: float, the ratio of average distance to select neighbor
    :param is_prob: bool, optional(default=True)
    :param with_feature: bool, optional(default=False)
    :return: instance of HyperG
    """
    assert isinstance(X, (np.ndarray, list))
    assert ratio > 0

    X = np.array(X)
    n_nodes = X.shape[0]
    n_edges = n_nodes

    m_dist = pairwise_distances(X)

    avg_dist = np.mean(m_dist)
    threshold = ratio * avg_dist

    coo = np.where(m_dist <= threshold)
    edge_idx, node_idx = coo

    if not is_prob:
        values = np.ones(node_idx.shape[0])
    else:
        m_neighbors_val = m_dist[coo]
        values = np.exp(-np.power(m_neighbors_val, 2.) / np.power(avg_dist, 2.))

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))

    w = np.ones(n_edges)

    if with_feature:
        return HyperG(H, w=w, X=X)

    return HyperG(H, w=w)

