import sys
import os
sys.path.append(os.path.abspath('../'))
import pyvista as pv
import numpy as np

def show_graph_and_mesh(V,E,R,mesh):
    p = pv.Plotter()
    mesh_actor = add_tet_mesh(p, mesh)
    lines_actor = p.add_lines(V[E].reshape(-1,3), color="k")
    sphere_actors = [None]*len(V)
    for i in range(len(V)):
        sphere_actors[i] = p.add_mesh(pv.Sphere(R[i], V[i]), color="b", opacity=0.3)


    def toggle_mesh_vis(flag) -> None:
        mesh_actor.SetVisibility(flag)

    def toggle_graph_vis(flag) -> None:
        lines_actor.SetVisibility(flag)
        for i in range(len(V)):
            sphere_actors[i].SetVisibility(flag)

    p.add_checkbox_button_widget(toggle_mesh_vis, value=True)
    p.add_checkbox_button_widget(toggle_graph_vis, value=True, position=(5.0, 0.0))
    p.show()

def show_graph_and_surf(V,E,R,surf):
    p = pv.Plotter()
    mesh_actor = add_tri_mesh(p, surf, show_edges=False)
    lines_actor = p.add_lines(V[E].reshape(-1,3), color="k")
    sphere_actors = [None]*len(V)
    for i in range(len(V)):
        sphere_actors[i] = p.add_mesh(pv.Sphere(R[i], V[i]), color="b", opacity=0.3)


    def toggle_mesh_vis(flag) -> None:
        mesh_actor.SetVisibility(flag)

    def toggle_graph_vis(flag) -> None:
        lines_actor.SetVisibility(flag)
        for i in range(len(V)):
            sphere_actors[i].SetVisibility(flag)

    p.add_checkbox_button_widget(toggle_mesh_vis, value=True)
    p.add_checkbox_button_widget(toggle_graph_vis, value=True, position=(5.0, 0.0))
    p.show()

def show_clusters(clusters):
    plotter = pv.Plotter()
    for i, cluster in enumerate(clusters):
        add_graph(plotter, cluster.V, cluster.E, cluster.R)
        plotter.add_point_labels(cluster.nodes[0].position, [str(i)])
    plotter.show()

def show_cluster_graph(cluster):
    show_graph(cluster.V, cluster.E, cluster.R)

def show_graph(V,E,R):
    plotter = pv.Plotter()
    plotter.add_lines(V[E].reshape(-1,3), color="k")
    points = pv.PolyData(V)
    points['R'] = R
    spheres = points.glyph(geom=pv.Sphere(radius=1.0), scale="R")
    plotter.add_mesh(spheres, opacity=0.5, color="b")
    plotter.show()

def add_graph(plotter, V,E,R):
    plotter.add_lines(V[E].reshape(-1,3), color="k")
    points = pv.PolyData(V)
    points['R'] = R
    spheres = points.glyph(geom=pv.Sphere(radius=1.0), scale="R")
    plotter.add_mesh(spheres, opacity=0.5, color="b")

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

    plotter = pv.Plotter()
    plotter.add_mesh(grid, color, lighting=True, show_edges=True)
    plotter.show()
def add_tri_mesh(plotter, tri_mesh, color='lightgrey', show_edges=True):
    tris = np.concatenate((np.full((tri_mesh.tris.shape[0], 1), 3), tri_mesh.tris), axis=1)
    grid = pv.PolyData(tri_mesh.nodes, tris)

    return plotter.add_mesh(grid, color, lighting=True, show_edges=show_edges)

def show_tet_mesh(tet_mesh, color='black'):
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tet_mesh.tets}, tet_mesh.nodes)
    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.add_mesh(grid.extract_all_edges(), color, lighting=True, show_edges=True)
    plotter.show()
def add_tet_mesh(plotter, tet_mesh, color='black'):
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tet_mesh.tets}, tet_mesh.nodes)
    plotter.add_axes()
    return plotter.add_mesh(grid.extract_all_edges(), color, lighting=True, show_edges=True)


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

