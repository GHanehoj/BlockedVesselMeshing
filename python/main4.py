from tools.mesh_util import load_tet, TetMesh
from tools.pyvista_plotting import show_tet_mesh
from rainbow.math.tetrahedron import *
from tqdm import tqdm
# import warp as wp
import pyvista as pv

msh = load_tet("small_4.mesh")
# points = wp.array3d(msh.nodes[msh.tets], dtype=float)
# r_outs = wp.array1d(np.zeros(len(msh.tets)), dtype=float)
# r_ins = wp.array1d(np.zeros(len(msh.tets)), dtype=float)
# wp.launch(compute_circumscribed_sphere, dim = len(msh.tets), inputs=[points, r_outs])
# wp.launch(compute_inscribed_sphere, dim = len(msh.tets), inputs=[points, r_ins])

# r_outs = r_outs.numpy()
# r_ins = r_ins.numpy()
r_outs = np.array([compute_circumscribed_sphere(*tet)[1] for tet in tqdm(msh.nodes[msh.tets])])
r_ins = np.array([compute_inscribed_sphere(*tet)[1] for tet in tqdm(msh.nodes[msh.tets])])
rad_ratio = 3*r_ins/r_outs

plotter = pv.Plotter()
plotter.add_axes()
grid = pv.UnstructuredGrid({pv.CellType.TETRA: msh.tets[rad_ratio > 0.01]}, msh.nodes)
plotter.add_mesh(grid.extract_all_edges(), 'grey', lighting=True, show_edges=True)
grid = pv.UnstructuredGrid({pv.CellType.TETRA: msh.tets[rad_ratio <= 0.01]}, msh.nodes)
plotter.add_mesh(grid.extract_all_edges(), 'red', lighting=True, show_edges=True, line_width=2)
plotter.show()
a=2