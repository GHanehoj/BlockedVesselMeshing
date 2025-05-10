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

def angle_in_plane(v1, v2s, n):
    v2s = np.asarray(v2s)
    scalar_input = False
    if v2s.ndim == 1:
        v2s = v2s[None,:]  # Makes x 1D
        scalar_input = True

    v1_p = v1 - np.dot(v1,n)*n
    v2_p = v2s - np.einsum("...i,i->...",v2s,n)[...,None]*n[None,...]
    ret = np.arctan2(np.dot(np.cross(v1_p, v2_p), n), np.einsum("i,...i->...", v1_p, v2_p))

    if scalar_input:
        return np.squeeze(ret)
    return ret

def angle_xy(p):
    return angle_in_plane(np.array([1.0, 0.0, 0.0]), p, np.array([0.0, 0.0, 1.0]))

def ang_mod(angs):
    return np.where(angs < -np.pi, angs + 2*np.pi,
           np.where(angs >  np.pi, angs - 2*np.pi,
                                   angs))

def rad_to_deg(rad):
    return rad/np.pi*180
def deg_to_rad(deg):
    return deg/180*np.pi

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)