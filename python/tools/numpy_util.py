import numpy as np

def normalize(v):
    return v / np.linalg.norm(v, axis=-1)

def lerp(v0, v1, t):
    assert(t >= 0 and t <= 1)
    return v0 + t*(v1-v0)

def mk_mask(idxs, tot):
    mask = np.full(tot, False)
    mask[idxs] = True
    return mask

def angle_in_plane(v1, v2, n):
    return np.arctan2(np.dot(np.cross(v1, v2), n), np.dot(v1, v2))

def angle_xy(p):
    return angle_in_plane(p, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))

def ang_mod(angs):
    return np.where(angs < -np.pi, angs + 2*np.pi, angs)