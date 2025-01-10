import sys
import os
sys.path.append(os.path.abspath('../'))
from skimage.measure import marching_cubes
from tools.mesh_util import TriMesh, calc_neigbours_tri
from tools.smoothing import taubin_smoothing
import convolution as CONV

def branch_contour(branch):
    V, E, R = branch.create_local_graph()
    grid = CONV.conv_surf(V, E, R)
    verts, tris = contour(grid)

    verts = branch.denormalize(verts)

    return verts, tris

def contour(grid):
    verts, tris, _, _ = marching_cubes(grid.values, 0)
    verts += grid.min
    verts *= grid.dx

    neighbours = calc_neigbours_tri(TriMesh(verts, tris))
    taubin_smoothing(verts, neighbours)

    return verts, tris
