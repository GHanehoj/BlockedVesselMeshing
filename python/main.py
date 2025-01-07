import numpy as np
from skimage.measure import marching_cubes
from convolution import conv_surf
from tools.smoothing import smooth_taubin
from tools.mesh_util import TriMesh
from tools.pyvista_plotting import show_tri_mesh

V = np.array([[0.0, 0, -20], [0, 0, 0], [10, 0, 20], [-5, 20, 30]])
E = np.array([[0,1], [1,2], [1,3]])
R = np.array([4, 2, 0.6, 1])

grid = conv_surf(V, E, R)

verts, faces, normals, values = marching_cubes(grid.values, 0)

verts2 = smooth_taubin(verts, faces)

mesh = TriMesh(verts2, faces)
show_tri_mesh(mesh)
a = 2
