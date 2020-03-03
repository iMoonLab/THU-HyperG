# coding=utf-8
import numpy as np
import scipy.sparse as sparse

from hyperg.base import HyperG


def test_hyperg():

    edge_idx = np.array([0, 0, 1, 1, 2, 2, 2])
    node_idx = np.array([0, 1, 2, 3, 0, 1, 4])
    val = np.array([0.1, 0.3, 0.2, 0.5, 0.6, 0.1, 0.3])

    H = sparse.coo_matrix((val, (node_idx, edge_idx)), shape=(5, 3))
    w = np.array([0.3, 0.4, 0.3])

    X = np.random.rand(5, 4)

    hg = HyperG(H, X=X, w=w)

    assert hg.num_edges() == 3
    assert hg.num_nodes() == 5
    assert np.allclose(hg.incident_matrix().A, H.A)
    assert np.allclose(hg.hyperedge_weights(), w)
    assert np.allclose(hg.node_features(), X)

    assert np.allclose(hg.node_degrees().data.reshape(-1), np.array([0.21, 0.12, 0.08, 0.2, 0.09]))
    assert np.allclose(hg.inv_square_node_degrees().data.reshape(-1),
                       np.array([2.1821789 , 2.88675135, 3.53553391, 2.23606798, 3.33333333]))
    assert np.allclose(hg.edge_degrees().data.reshape(-1), np.array([0.4, 0.7, 1.0]))
    assert np.allclose(hg.inv_edge_degrees().data.reshape(-1), np.array([1/0.4, 1/0.7, 1/1.0]))

    DV2 = hg.inv_square_node_degrees()
    INVDE = hg.inv_edge_degrees()
    THETA = DV2.dot(H).dot(sparse.diags(w)).dot(INVDE).dot(H.T).dot(DV2)

    assert np.allclose(hg.theta_matrix().A, THETA.A)
    assert np.allclose(hg.laplacian().A, (sparse.eye(5) - THETA).A)

    hg.update_hyedge_weights(np.array([1.0, 1.0, 1.0]))
    assert np.allclose(hg.hyperedge_weights(), np.array([1.0, 1.0, 1.0]))
    assert np.allclose(hg.node_degrees().data.reshape(-1), np.array([0.7, 0.4, 0.2, 0.5, 0.3]))

    edge_idx = np.array([0, 1, 1, 2, 2, 2])
    node_idx = np.array([0, 2, 3, 0, 1, 4])
    val = np.array([0.2, 0.4, 0.5, 0.6, 0.1, 0.3])

    H = sparse.coo_matrix((val, (node_idx, edge_idx)), shape=(5, 3))
    hg.update_incident_matrix(H)
    assert np.allclose(hg.incident_matrix().A, H.A)
    assert np.allclose(hg.node_degrees().data.reshape(-1), np.array([0.8, 0.1, 0.4, 0.5, 0.3]))













if __name__ == "__main__":
    test_hyperg()
