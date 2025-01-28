import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import data as DATA
import matplotlib.pyplot as plt
from tools.numpy_util import angle_between

def compute_neighbour_matrix(V, E):
    """
    Computes a neighbour matrix where the i'th entry contains
    the indices of the neighbours of vertex i. Since all nodes
    are of cardinality <4, this can be stored in (Nx3).
    Missing neighbours are represented as -1.
    """

    max_N = np.max(np.unique(E, return_counts=True)[1])
    neighbours = np.full((len(V), max_N), -1, dtype=np.int32)
    for e in E:
        n0 = neighbours[e[0]]
        n0[np.sum(n0 != -1)] = e[1]
        n1 = neighbours[e[1]]
        n1[np.sum(n1 != -1)] = e[0]
    return neighbours


tree_folder = f"../../data/trees/-reg500"
V, E, R = DATA.load_skeleton_data(tree_folder)

neighbours = compute_neighbour_matrix(V, E)

internal_mask = np.sum(neighbours != -1, axis=1) > 1


centers = V[neighbours[internal_mask]]
D = np.mean(centers, axis=1) - V[internal_mask]

arms = centers-V[internal_mask, None, :]
angs = angle_between(arms[:,:,None,:], arms[:,None,:,:])
angs[:, np.tril_indices(5)[0], np.tril_indices(5)[1]] = np.nan

lens = np.linalg.norm(arms, axis=2)/R[internal_mask, None]

plt.hist(angs.flatten(), bins=100)
plt.show()

plt.hist(lens[lens < 5].flatten(), bins=100)
plt.show()
