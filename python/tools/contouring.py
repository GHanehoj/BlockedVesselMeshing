import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
from skimage.measure import marching_cubes
from tools.mesh_util import TriMesh
from tools.smoothing import taubin_smoothing_nomask
import convolution as CONV

def cluster_contour(cluster, res):
    return graph_contour(cluster.V, cluster.E, cluster.R, res)

def graph_contour(V, E, R, res):
    dx = (2*np.min(R))/res
    sz = 2/np.median(R)*0.5 # Convolution surface math is not scale-independent, thus we temporarily rescale to fixed size.
    grid = CONV.conv_surf(V*sz, E, R*sz, dx*sz)
    verts, tris = contour(grid)
    verts /= sz
    return verts, tris

def contour(grid):
    verts, tris, _, _ = marching_cubes(grid.values, 0, method="lorensen")
    verts += grid.min
    verts *= grid.dx

    neighbours = TriMesh(verts, tris).calc_neighbours()
    taubin_smoothing_nomask(verts, neighbours, iter=20)

    return verts, tris
