import sys
import os
sys.path.append(os.path.abspath('../'))
import pyvista as pv
import numpy as np

def show_seg_mesh(seg_mesh, color='black'):
    show_seg_meshes([seg_mesh], color)
def show_seg_meshes(seg_meshes, color='black'):
    plotter = pv.Plotter()
    for mesh in seg_meshes:
        grid = pv.UnstructuredGrid({pv.CellType.LINE: mesh.segs}, mesh.nodes)
        plotter.add_mesh(grid, color, lighting=True, show_edges=True)
    plotter.show()

def show_tri_mesh(tri_mesh, color='lightgrey'):
    tris = np.concatenate((np.full((tri_mesh.tris.shape[0], 1), 3), tri_mesh.tris), axis=1)
    grid = pv.PolyData(tri_mesh.nodes, tris)
    # grid2 = grid.smooth_taubin()

    plotter = pv.Plotter()
    # plotter.add_mesh(grid, color, lighting=True, show_edges=True, opacity=0.5)
    plotter.add_mesh(grid, color, lighting=True, show_edges=True)
    plotter.show()

def show_tet_mesh(tet_mesh, color='black'):
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tet_mesh.tets}, tet_mesh.nodes)
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.add_mesh(grid.extract_all_edges(), color, lighting=True, show_edges=True)
    plotter.show()

def add_seg_mesh(plotter, seg_mesh):
    grid = pv.UnstructuredGrid({pv.CellType.LINE: seg_mesh.segs}, seg_mesh.nodes)
    plotter.add_mesh(grid, 'black', lighting=True, show_edges=True)
def add_tri_mesh(plotter, tri_mesh):
    grid = pv.UnstructuredGrid({pv.CellType.TRIANGLE: tri_mesh.tris}, tri_mesh.nodes)
    plotter.add_mesh(grid, 'lightgrey', lighting=True, show_edges=True)
def add_tet_mesh(plotter, tet_mesh):
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tet_mesh.tets}, tet_mesh.nodes)
    plotter.add_mesh(grid, 'lightgrey', lighting=True, show_edges=True)



# plotter = pv.Plotter()
# plotter.add_mesh(grid1, 'lightgrey', lighting=True, show_edges=True, opacity=0.5)
# plotter.add_mesh(grid2, 'lightgrey', lighting=True, show_edges=True, opacity=0.5)
# plotter.show()

# plotter = pv.Plotter()
# plotter.add_mesh(ends1[arm], 'red', lighting=True, show_edges=True)
# plotter.add_mesh(strip1, 'blue', lighting=True, show_edges=True)
# plotter.add_mesh(connector, 'green', lighting=True, show_edges=True)
# plotter.add_mesh(strip2, 'blue', lighting=True, show_edges=True)
# plotter.add_mesh(ends2[0], 'red', lighting=True, show_edges=True)
# plotter.show()

# plotter = pv.Plotter()
# plotter.add_mesh(grid1.extract_all_edges(), color="k", lighting=True, show_edges=True, opacity=0.5)
# plotter.add_mesh(grid2.extract_all_edges(), color="k", lighting=True, show_edges=True, opacity=0.5)
# plotter.show()

