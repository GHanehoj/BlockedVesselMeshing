import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import pyvista as pv
from tools.mesh_util import merge_duplicate_nodes

def contour(grid):
    pvg = pv.ImageData(dimensions=grid.dim, spacing=[grid.dx]*3, origin=grid.min*grid.dx)
    msh = pvg.contour([0], np.transpose(grid.values, [2,1,0]).flatten(), method="marching_cubes")
    smth = msh.smooth_taubin()
    verts = smth.points
    tris = smth.faces.reshape(-1,4)[:,1:]

    verts, tris = merge_duplicate_nodes(verts, tris, tol=0.01)
    # verts, tris, _, _ = marching_cubes(grid.values, 0, method="lorensen")
    # verts += grid.min
    # verts *= grid.dx

    # neighbours = TriMesh(verts, tris).calc_neighbours()
    # taubin_smoothing_nomask(verts, neighbours, iter=50)

    return verts, tris
