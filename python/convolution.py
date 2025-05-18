"""
This module implements convolution surfaces.
A method for converting a skeleton structure into a surface mesh.
"""
import data as DATA
import kernel as KERNEL
import render as RENDER
import numpy as np
def conv_surf(V, E, R, dx):
    render_data = DATA.RenderData(V, E, R, dx)
    if np.prod(render_data.dim) > 200*200*200:
        raise Exception("convolution too large")
    kernel, _, iso_value = KERNEL.create_kernel(kernel_type="oeltze.preim")
    grid, _ = RENDER.render_field(dx=dx
                                , iso_value=iso_value
                                , data=render_data
                                , kernel=kernel
                                , verbose=False
                                )
    return grid
