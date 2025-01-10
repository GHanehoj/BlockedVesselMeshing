"""
This module implements convolution surfaces.
A method for converting a skeleton structure into a surface mesh.
"""
import tools.resolution_util as RES
import data as DATA
import kernel as KERNEL
import render as RENDER

def conv_surf(V, E, R, res=25):
    dx = RES.get_resolution(V, E, R, res)
    render_data = DATA.RenderData(V, E, R, dx)
    kernel, _, iso_value = KERNEL.create_kernel(kernel_type="oeltze.preim")
    grid, _ = RENDER.render_field(dx=dx
                                , iso_value=iso_value
                                , data=render_data
                                , kernel=kernel
                                , verbose=False
                                )
    return grid
