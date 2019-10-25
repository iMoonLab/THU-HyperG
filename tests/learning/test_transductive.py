import numpy as np
import scipy.sparse as sparse

from thumoon.learning import *
from thumoon.base import HyperG


def test_trans_infer():
    edge_idx = np.array([0, 0, 1, 1, 2, 2, 2])
    node_idx = np.array([0, 1, 2, 3, 0, 1, 4])
    val = np.array([0.1, 0.3, 0.2, 0.5, 0.6, 0.1, 0.3])

    H = sparse.coo_matrix((val, (node_idx, edge_idx)), shape=(5, 3))
    hg = HyperG(H)

    y = np.array([0, 1, 1, -1, -1])

    y_predict = trans_infer(hg, y, lbd=100)

    assert y_predict.shape[0] == 2


if __name__ == "__main__":
    test_trans_infer()