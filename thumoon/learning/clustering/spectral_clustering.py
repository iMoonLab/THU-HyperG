# coding=utf-8
import numpy as np
from sklearn.cluster import k_means

from thumoon.base import HyperG


def spectral_hg_partitioning(hg, n_clusters, n_components=None, n_init=10):
    """
    :param hg: instance of HyperG
    :param n_clusters: int,
    :param n_components: int,
    :param n_init: int, number of time the k-means algorithm will be run
    with different centroid seeds.
    :return: numpy array, shape = (n_samples,), labels of each point
    """

    assert isinstance(hg, HyperG)
    assert n_clusters <= hg.num_nodes()

    if n_components is None:
        n_components = n_clusters

    L = hg.laplacian()
    eigenval, eigenvec = np.linalg.eig(L.toarray())

    idxs = np.argsort(eigenval)[:n_components]
    embeddings = eigenvec[:, idxs]

    _, labels, _ = k_means(embeddings, n_clusters, n_init=n_init)
    return labels
