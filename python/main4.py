import convolution as CONV
from tools.pyvista_plotting import add_graph, add_tet_mesh
from tools.mesh_util import TriMesh, load_tet
import numpy as np
import pyvista as pv

np.seterr(divide="raise", invalid="raise")

# mesh = load_tet("/media/data/data/meshes/meshes/brain_vg_3.mesh")
# mesh = load_tet("/media/data/data/meshes/meshes/lung_4.mesh")
mesh = load_tet("/media/data/data/meshes/meshes/tree_ahn_3.mesh")

plotter = pv.Plotter()
add_tet_mesh(plotter, mesh)
plotter.show()