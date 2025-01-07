import numpy as np

def get_resolution(V, E, R, max_res):
    padding = 4*np.max(R)
    min_V = np.min(V, axis=0)
    max_V = np.max(V, axis=0)
    sz = np.max(max_V-min_V+padding)
    dx = sz/max_res
    return dx
