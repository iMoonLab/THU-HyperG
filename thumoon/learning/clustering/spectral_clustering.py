# coding=utf-8
from sklearn.cluster import k_means
from sklearn.cluster.spectral import discretize
from sklearn.utils import check_symmetric, check_random_state
from scipy.linalg import eigh

from thumoon.base import HyperG


def spectral_hg_partitioning(hg, n_clusters, assign_labels='kmeans', n_components=None, random_state=None, n_init=10):
    """
    :param hg: instance of HyperG
    :param n_clusters: int,
    :param assign_labels: str, {'kmeans', 'discretize'}, default: 'kmeans'
    :param n_components: int, number of eigen vectors to use for the spectral embedding
    :param random_state: int or None (default)
    :param n_init: int, number of time the k-means algorithm will be run
    with different centroid seeds.
    :return: numpy array, shape = (n_samples,), labels of each point
    """

    assert isinstance(hg, HyperG)
    assert n_clusters <= hg.num_nodes()

    random_state = check_random_state(random_state)

    if n_components is None:
        n_components = n_clusters

    L = hg.laplacian().toarray()
    L = check_symmetric(L)

    eigenval, eigenvec = eigh(L)
    embeddings = eigenvec[:, :n_components]

    if assign_labels == 'kmeans':
        _, labels, _ = k_means(embeddings, n_clusters, random_state=random_state,
                               n_init=n_init)
    else:
        labels = discretize(embeddings, random_state=random_state)

    return labels
