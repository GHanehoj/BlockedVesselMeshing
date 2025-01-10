import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np

def laplacian_smoothing(V, neighbours, fixed_mask, lr=0.1, iter=50):
    if fixed_mask is None: fixed_mask = np.full(len(V), False)
    neighbour_mask = np.isclose(np.sum(neighbours[~fixed_mask], axis=1), 1.0)
    mask = ~fixed_mask
    mask[~fixed_mask] = neighbour_mask
    for _ in range(iter):
        d = neighbours@V - V
        V[mask] += lr * d[mask]

def taubin_smoothing(V, neighbours, fixed_mask = None, lamb = 0.5, nu = 0.5, iter=50):
    if fixed_mask is None: fixed_mask = np.full(len(V), False)
    neighbour_mask = np.isclose(np.sum(neighbours[~fixed_mask], axis=1), 1.0)
    mask = ~fixed_mask
    mask[~fixed_mask] = neighbour_mask
    for i in range(iter):
        d = neighbours@V - V
        if i % 2 == 0:
            V[mask] += lamb * d[mask]
        else:
            V[mask] -= nu * d[mask]
