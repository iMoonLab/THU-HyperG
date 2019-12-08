import numpy as np
from thumoon.generation import concat_multi_hg, gen_knn_hg, gen_clustering_hg


def test_fuse_multi_hg():
    X = np.random.rand(200, 10)
    knn_hg = gen_knn_hg(X, n_neighbors=5)
    clu_hg = gen_clustering_hg(X, n_clusters=20)

    hg = concat_multi_hg([knn_hg, clu_hg])

    assert hg.num_nodes() == knn_hg.num_nodes()
    assert hg.num_nodes() == clu_hg.num_nodes()
    assert hg.num_edges() == knn_hg.num_edges() + clu_hg.num_edges()

