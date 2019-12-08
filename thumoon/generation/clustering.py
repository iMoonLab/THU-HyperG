# coding=utf-8
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import scipy.sparse as sparse

from thumoon.base import HyperG
from thumoon.utils import print_log


def gen_clustering_hg(X, n_clusters, method="kmeans", with_feature=False, random_state=None):
    """
    :param X: numpy array, shape = (n_samples, n_features)
    :param n_clusters: int, number of clusters
    :param method: str, clustering methods("kmeans",)
    :param with_feature: bool, optional(default=False)
    :param random_state: int, optional(default=False) determines random number generation
    for centroid initialization
    :return: instance of HyperG
    """
    if method == "kmeans":
        cluster = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X).labels_
    else:
        raise ValueError("{} method is not supported".format(method))

    assert n_clusters >= 1

    n_edges = n_clusters
    n_nodes = X.shape[0]

    node_idx = np.arange(n_nodes)
    edge_idx = cluster

    values = np.ones(node_idx.shape[0])
    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))

    w = np.ones(n_edges)

    if with_feature:
        return HyperG(H, w=w, X=X)

    return HyperG(H, w=w)
