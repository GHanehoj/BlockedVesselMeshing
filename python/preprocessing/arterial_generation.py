import sys
import os
sys.path.append(os.path.abspath('../'))
import tree as TREE
import numpy as np
import rainbow.math.vector3 as VEC

def generate_arterial(V, E, R):
    _, nodes = TREE.make_tree(V, E, R)

    leaf_mask = np.array([len(node.children) == 0 for node in nodes])
    edge_mask = ~leaf_mask[E]
    offsets = np.cumsum(leaf_mask)
    displacements = VEC.unit(np.random.random((np.sum(~leaf_mask),3)))

    V2 = V[~leaf_mask] + 3.5*R[~leaf_mask, None]*displacements
    E2 = E.copy()
    E2[edge_mask] += V.shape[0]
    E2[edge_mask] -= offsets[E][edge_mask]
    R2 = R[~leaf_mask]

    tot_V = np.vstack((V, V2))
    tot_E = np.vstack((E, E2))
    tot_R = np.hstack((R, R2))

    # Mask. 0 = Vein root, 1 = Vein branch, 2 = leaf, 3 = Arterial branch, 4 = Arterial root
    mask = np.empty(tot_V.shape[0])
    mask[np.where(leaf_mask)] = 2
    mask[np.where(~leaf_mask)] = 1
    mask[V.shape[0]:] = 3
    mask[0] = 0
    mask[V.shape[0]] = 4

    return tot_V, tot_E, tot_R, mask
