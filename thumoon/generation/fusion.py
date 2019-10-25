# coding=utf-8
import numpy as np
import scipy.sparse as sparse

from thumoon.base import HyperG


def concat_multi_hg(hg_list):
    """concatenate multiple hypergraphs to one hypergraph
    :param hg_list: list, list of HyperG instance
    :return: instance of HyperG
    """
    H_s = [hg.incident_matrix() for hg in hg_list]
    w_s = [hg.hyperedge_weights() for hg in hg_list]

    H = sparse.hstack(H_s)
    w = np.hstack(w_s)

    X = None
    for hg in hg_list:
        if X is not None and hg.node_features() is not None:
            assert X == hg.node_features()
        elif hg.node_features() is not None:
            X = hg.node_features()

    return HyperG(H, X=X, w=w)


def fuse_mutli_sub_hg(hg_list):
    """
    :param hg_list: list, list of HyperG instance
    :return: instance of HyperG
    """
    incident_mat_row = [hg.incident_matrix().row for hg in hg_list]
    incident_mat_col = [hg.incident_matrix().col for hg in hg_list]
    incident_mat_data = [hg.incident_matrix().data for hg in hg_list]

    num_nodes = [hg.num_nodes() for hg in hg_list]
    num_edges = [hg.num_edges() for hg in hg_list]

    nodes_to_add = [0] + [sum(num_nodes[:i+1]) for i in range(len(hg_list)-1)]
    edges_to_add = [0] + [sum(num_edges[:i+1]) for i in range(len(hg_list)-1)]

    for i in range(len(hg_list)):
        incident_mat_row[i] = incident_mat_row[i] + nodes_to_add[i]
        incident_mat_col[i] = incident_mat_col[i] + edges_to_add[i]

    incident_mat_row = np.concatenate(incident_mat_row)
    incident_mat_col = np.concatenate(incident_mat_col)
    incident_mat_data = np.concatenate(incident_mat_data)

    H = sparse.coo_matrix((incident_mat_data, (incident_mat_row, incident_mat_col)),
                          shape=(sum(num_nodes), sum(num_edges)))

    return HyperG(H)
