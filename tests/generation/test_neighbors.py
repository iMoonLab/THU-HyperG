# coding=utf-8
import numpy as np

from hyperg.generation import gen_knn_hg, gen_epsilon_ball_hg


def test_gen_knn_hg():
    X = np.random.rand(50, 100)
    hg = gen_knn_hg(X, n_neighbors=5)

    assert hg.num_nodes() == 50
    assert hg.num_edges() == 50
    assert hg.incident_matrix().nnz == 300


def test_gen_epsilon_ball_hg():
    X = np.random.rand(50, 100)
    hg = gen_epsilon_ball_hg(X, ratio=0.1)
    assert hg.num_nodes() == 50
    assert hg.num_edges() == 50
