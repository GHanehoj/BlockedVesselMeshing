import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
from skimage.measure import marching_cubes
from tools.mesh_util import TriMesh
from grid import Grid
import convolution as CONV

def branch_contour(branch):
    V, E, R = branch.create_local_graph()
    grid = CONV.conv_surf(V, E, R)
    verts, tris = contour(grid)

    verts = branch.denormalize(verts)

    return verts, tris

def contour(grid: Grid) -> TriMesh:
    verts, tris, _, _ = marching_cubes(grid.values, 0)
    verts += grid.min
    verts *= grid.dx

    verts = smooth_taubin(verts, tris)

    return verts, tris


def smooth_taubin(verts, tris, lamb=0.5, nu=0.5, iterations=20):
    verts2 = verts.copy()
    neighbours = mk_neigbours(verts2, tris)

    for i in range(iterations):
        dot = neighbours.dot(verts2) - verts2
        if i % 2 == 0:
            verts2 += lamb * dot
        else:
            verts2 -= nu * dot

    return verts2


def mk_neigbours(verts, tris):
    L = np.zeros((len(verts), len(verts)))
    edge_idxs = [[0,1], [1,2], [2,0]]
    for t in tris:
        for e0, e1 in t[edge_idxs]:
            L[e0,e1] += 0.5
            L[e1,e0] += 0.5
    tots = np.sum(L, axis=1)
    L = L/tots[:, None]
    return L
