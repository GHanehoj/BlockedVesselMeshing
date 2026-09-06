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
    R *= 0.65 # Simple convolution surface radius is not consistent. At normalized scales, 0.65 is tuned to make radius actually match
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

def conv_surf_SCALIS(V, E, R, dx):
    render_data = RENDER.RenderData(V, E, R, dx)
    if np.prod(render_data.dim) > 250*250*250:
        raise Exception("convolution too large")
    grid = RENDER.render_field_SCALIS(dx=dx
                      , iso_value=1
                      , data=render_data
                      , sigma=1.1
                      )
    return grid


def contour(grid):
    pvGrid = pv.ImageData(dimensions=grid.dim, spacing=[grid.dx]*3, origin=grid.min*grid.dx)
    mesh = pvGrid.contour([0], np.transpose(grid.values, [2,1,0]).flatten(), method="marching_cubes")
    mesh = mesh.smooth_taubin()
    verts = mesh.points
    tris = mesh.faces.reshape(-1,4)[:,1:]

    verts, tris = merge_duplicate_nodes(verts, tris, tol=0.01)

    verts, tris = pymeshfix.clean_from_arrays(verts, tris)

    return verts, tris

