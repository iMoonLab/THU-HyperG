import numpy as np
from thumoon.generation import gen_grid_neigh_hg


def test_gen_grid_neigh_hg():
    grid = np.random.rand(5,4)
    hg = gen_grid_neigh_hg(grid.shape)

    assert hg.num_nodes() == 20
    assert hg.num_edges() == 20

    H = hg.incident_matrix()
    assert np.allclose(H.data, np.ones_like(H.data))
