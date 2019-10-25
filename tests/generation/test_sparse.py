# coding=utf-8
import numpy as np

from thumoon.generation import gen_l1_hg


def test_gen_l1_hg():
    X = np.random.rand(50, 100)
    hg = gen_l1_hg(X, gamma=1., n_neighbors=5, log=False)

    assert hg.num_nodes() == 50
    assert hg.num_edges() == 50
    assert hg.incident_matrix().nnz == 300