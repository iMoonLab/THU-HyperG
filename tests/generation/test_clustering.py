# coding=utf-8
import numpy as np
from thumoon.generation import gen_clustering_hg


def test_gen_clus_hg():
    X = np.random.rand(50, 100)
    hg = gen_clustering_hg(X, n_clusters=10)

    assert hg.num_nodes() == 50
    assert hg.num_edges() == 10
    assert hg.incident_matrix().nnz == 50
