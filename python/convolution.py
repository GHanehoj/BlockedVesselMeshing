"""
This module implements convolution surfaces.
A method for converting a skeleton structure into a surface mesh.
"""
import kernel as KERNEL
import render as RENDER
from tools.mesh_util import merge_duplicate_nodes
import numpy as np
import pyvista as pv
import pymeshfix

def conv_surf(V, E, R, dx):
    render_data = RENDER.RenderData(V, E, R, dx)
    if np.prod(render_data.dim) > 250*250*250:
        raise Exception("convolution too large")
    kernel, _, iso_value = KERNEL.create_kernel(kernel_type="oeltze.preim")
    grid = RENDER.render_field(dx=dx
                      , iso_value=iso_value
                      , data=render_data
                      , kernel=kernel
                      )
    return grid


def contour(grid):
    pvg = pv.ImageData(dimensions=grid.dim, spacing=[grid.dx]*3, origin=grid.min*grid.dx)
    msh = pvg.contour([0], np.transpose(grid.values, [2,1,0]).flatten(), method="marching_cubes")
    smth = msh.smooth_taubin()
    verts = smth.points
    tris = smth.faces.reshape(-1,4)[:,1:]

    verts, tris = merge_duplicate_nodes(verts, tris, tol=0.01)

    verts, tris = pymeshfix.clean_from_arrays(verts, tris)

    return verts, tris

