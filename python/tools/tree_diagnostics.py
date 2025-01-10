import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import data as DATA
import matplotlib.pyplot as plt
def compute_neigbor_matrix(V, E):
    """
    Computes a neighbor matrix where the i'th entry contains
    the indices of the neighbors of vertex i. Since all nodes
    are of cardinality <4, this can be stored in (Nx3).
    Missing neighbors are represented as -1.
    """

    neighbors = np.empty((len(V), 3), dtype=np.int32)
    neighbors[:] = -1
    for e in E:
        n0 = neighbors[e[0]]
        n0[np.sum(n0 != -1)] = e[1]
        n1 = neighbors[e[1]]
        n1[np.sum(n1 != -1)] = e[0]
    return neighbors

def angle_between(v1, v2):
    v1_u = v1/np.linalg.norm(v1, axis=-1)[...,None]
    v2_u = v2/np.linalg.norm(v2, axis=-1)[...,None]
    return np.arccos(np.clip(np.einsum("...k,...k->...",v1_u, v2_u), -1.0, 1.0))


tree_folder = f"../../data/trees/reg150"
V, E, R = DATA.load_skeleton_data(tree_folder)

neighbors = compute_neigbor_matrix(V, E)

internal_mask = np.sum(neighbors != -1, axis=1) == 3

centers = V[neighbors[internal_mask]]
D = np.mean(centers, axis=1) - V[internal_mask]

arms = centers-V[internal_mask, None, :]
idxs = np.array([[0,1], [1,2], [2,0]])
angs = angle_between(arms[:, idxs[:,0]], arms[:, idxs[:,1]])

lens = np.linalg.norm(arms, axis=2)/R[internal_mask, None]

# plt.hist(angs)
# plt.show()

plt.hist(lens[lens < 5], bins=100)
plt.show()
