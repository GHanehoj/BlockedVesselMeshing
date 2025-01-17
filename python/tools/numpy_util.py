import numpy as np

def normalize(v):
    return v / np.linalg.norm(v, axis=-1)[...,None]

def lerp(v0, v1, t):
    return v0 + t*(v1-v0)

def mk_mask(idxs, tot):
    idxs = np.array(idxs)
    if idxs.ndim == 0: idxs = np.array([idxs])
    mask = np.full(tot, False)
    if len(idxs) > 0:
        mask[idxs] = True
    return mask

def angle_between(v1, v2):
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return np.arccos(np.clip(np.einsum("...k,...k->...",v1_u, v2_u), -1.0, 1.0))

def angle_in_plane(v1, v2, n):
    return np.arctan2(np.dot(np.cross(v1, v2), n), np.dot(v1, v2))

def angle_xy(p):
    return angle_in_plane(p, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))

def ang_mod(angs):
    return np.where(angs < -np.pi, angs + 2*np.pi,
           np.where(angs >  np.pi, angs - 2*np.pi,
                                   angs))