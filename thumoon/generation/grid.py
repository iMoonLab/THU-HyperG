# coding=utf-8
import numpy as np
import scipy.sparse as sparse

from thumoon.base import HyperG


def gen_grid_neigh_hg(input_size):
    """
    :param input_size: numpy array, shape = (2, ), (height, width)
    :return: instance of HyperG
    """
    input_size = np.array(input_size).reshape(-1)
    assert input_size.shape[0] == 2

    # TODO
    h, w = input_size
    n_nodes = w * h

    node_set = np.arange(n_nodes)

    neigh_idx = [
        node_set - w - 1,
        node_set - w,
        node_set - w + 1,

        node_set - 1,
        node_set,
        node_set + 1,

        node_set + w - 1,
        node_set + w,
        node_set + w + 1,
    ]

    neigh_mask = [
        (node_set // w == 0) | (node_set % w == 0),
        (node_set // w == 0),
        (node_set // w == 0) | (node_set % w == w - 1),

        (node_set % w == 0),
        np.zeros_like(node_set, dtype=np.bool),
        (node_set % w == w - 1),

        (node_set // w == h-1) | (node_set % w == 0),
        (node_set // w == h-1),
        (node_set // w == h-1) | (node_set % w == w - 1),
    ]

    # mask
    for i in range(len(neigh_idx)):
        neigh_idx[i][neigh_mask[i]] = -1

    node_idx = np.hstack(neigh_idx)
    edge_idx = np.tile(node_set.reshape(1, -1), [len(neigh_idx), 1]).reshape(-1)
    values = np.ones_like(node_idx)

    # filter negative elements
    non_neg_idx = np.where(node_idx != -1)

    node_idx = node_idx[non_neg_idx]
    edge_idx = edge_idx[non_neg_idx]
    values = values[non_neg_idx]

    n_edges = n_nodes
    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))

    return HyperG(H)


if __name__ == "__main__":
    gen_grid_neigh_hg((4, 5))