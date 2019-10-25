# coding=utf-8
import numpy as np
from sklearn.cluster import KMeans
import scipy.sparse as sparse

from thumoon.base import HyperG
from thumoon.utils import print_log


def gen_clustering_hg(X, n_edges, method="kmeans", with_feature=False, random_state=None):
    """
    :param X: numpy array, shape = (n_samples, n_features)
    :param n_edges: int, number of hyperedges
    :param method: str, clustering methods("kmeans",)
    :param with_feature: bool, optional(default=False)
    :param random_state: int, optional(default=False) determines random number generation
    for centroid initialization
    :return: instance of HyperG
    """
    # TODO: 1: is_prob, 2: hyperedge weights
    if method == "kmeans":
        cluster = KMeans(n_clusters=n_edges, random_state=random_state).fit(X).labels_
    else:
        raise ValueError("{} method is not supported".format(method))

    assert n_edges >= 1

    n_nodes = X.shape[0]

    node_idx = np.arange(n_nodes)
    edge_idx = cluster

    values = np.ones(node_idx.shape[0])
    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))

    if with_feature:
        return HyperG(H, X=X)
    else:
        return HyperG(H)