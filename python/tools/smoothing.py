import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np

def laplacian_smoothing(V, neighbours, fixed_mask, lr=0.1, iter=50):
    if fixed_mask is None: fixed_mask = np.full(len(V), False)
    connected_mask = np.isclose(np.sum(neighbours[~fixed_mask], axis=1), 1.0)
    mask = ~fixed_mask
    mask[~fixed_mask] = connected_mask
    for _ in range(iter):
        d = neighbours[mask]@V - V[mask]
        V[mask] += lr * d

def taubin_smoothing(V, neighbours, fixed_mask = None, lamb = 0.5, nu = 0.5, iter=50):
    if fixed_mask is None: fixed_mask = np.full(len(V), False)
    connected_mask = np.isclose(np.sum(neighbours[~fixed_mask], axis=1), 1.0)
    mask = ~fixed_mask
    mask[~fixed_mask] = connected_mask
    for i in range(iter):
        d = neighbours[mask]@V - V[mask]
        if i % 2 == 0:
            V[mask] += lamb * d
        else:
            V[mask] -= nu * d

def taubin_smoothing_nomask(V, neighbours, lamb = 0.5, nu = 0.5, iter=50):
    for i in range(iter):
        d = neighbours@V - V
        if i % 2 == 0:
            V += lamb * d
        else:
            V -= nu * d
