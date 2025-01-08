import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import convolution as CONV
from tools.contouring import contour
import branch as BRANCH
from tree import TreeConf
import rainbow.math.quaternion as QUAT
import rainbow.math.vector3 as VEC
import rainbow.math.angle as ANG
import tetgen
import igl
import pyvista as pv
from tools.mesh_util import SegMesh, TriMesh, TetMesh, merge_tri_meshes, merge_tet_meshes
from tools.pyvista_plotting import show_tet_mesh, show_tri_mesh, show_seg_meshes, add_seg_mesh, add_tri_mesh

V_proto = np.array([[ 0.0,            0.0, -1.0           ],
                    [ 0.0,            0.0,  0.0           ],
                    [ 0.5*np.sqrt(2), 0.0,  0.5*np.sqrt(2)],
                    [-0.5*np.sqrt(2), 0.0,  0.5*np.sqrt(2)]])
E_proto = np.array([[0,1], [1,2], [1,3]])
R_proto = np.array([1/3, 1/3, 1/3, 1/3])

def normalize(v):
    return v / np.linalg.norm(v, axis=-1)

def gen_proto():
    grid = CONV.conv_surf(V_proto, E_proto, R_proto)
    v, t = contour(grid)
    tgen = tetgen.TetGen(v, t)
    nodes, elems = tgen.tetrahedralize()
    return nodes, np.int64(elems)

def remove_tip(nodes, cells, point, dir):
    mask = np.dot(nodes-point, dir) > 0
    beyond_idxs = np.where(mask)[0]

    cells_mask = np.any(cells[:,:,None] == beyond_idxs, axis=2)

    single_pruned_cells = (np.sum(cells_mask, axis=1) == 0)

    cells_pruned = cells[single_pruned_cells]

    return cells_pruned


def biharmonic_templating(t, arm_dir):
    v, f = gen_proto()
    u = v.copy()

    s = np.full((v.shape[0], 1), -1)
    s[np.dot(v-t*V_proto[0], V_proto[0]) > 0] = 0
    s[np.dot(v-t*V_proto[2], V_proto[2]) > 0] = 1
    s[np.dot(v-t*V_proto[3], V_proto[3]) > 0] = 2

    b = np.array([[t[0] for t in [(i, s[i]) for i in range(0, v.shape[0])] if t[1] >= 0]]).T

    ## Boundary conditions directly on deformed positions
    u_bc = np.zeros((b.shape[0], v.shape[1]))
    v_bc = np.zeros((b.shape[0], v.shape[1]))

    for bi in range(b.shape[0]):
        v_bc[bi] = v[b[bi]]
        if s[b[bi]] == 1: # Move handle 1 over
            Q = QUAT.R_vector_to_vector(V_proto[2], arm_dir)
            u_bc[bi] = QUAT.rotate_array(Q, v[b[bi]])
        else: # Move other handles forward
            u_bc[bi] = v[b[bi]]

    d_bc = u_bc - v_bc
    d = igl.harmonic(v, f, b, d_bc, 1)
    u = v + d
    return TetMesh(u,f), s


def arap_templating(t, arm_dir):
    v, f = gen_proto()

    s = np.full((v.shape[0], 1), -1)
    s[np.dot(v-t*V_proto[0], V_proto[0]) > 0] = 0
    s[np.dot(v-t*V_proto[2], V_proto[2]) > 0] = 1
    s[np.dot(v-t*V_proto[3], V_proto[3]) > 0] = 2

    # Vertices in selection
    b = np.array([[t[0] for t in [(i, s[i]) for i in range(0, v.shape[0])] 
        if t[1] >= 0]]).T

    # Precomputation
    arap = igl.ARAP(v, f, 3, b)

    bc = np.zeros((b.size, v.shape[1]))
    for i in range(0, b.size):
        bc[i] = v[b[i]]
        if s[b[i]] == 1:
            Q = QUAT.R_vector_to_vector(V_proto[2], arm_dir)
            bc[i] = QUAT.rotate_array(Q, v[b[i]])
    vn = arap.solve(bc, v)
    return TetMesh(vn,f), s

def show_deform(plotter, tet_mesh, s, arm_dir):
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: tet_mesh.tets}, tet_mesh.nodes)
    plotter.add_axes()
    # plotter.add_mesh(grid.extract_all_edges(), "black", lighting=True, show_edges=True)
    plotter.add_mesh(grid, scalars=s+1, cmap=["#929591", "#dbb40c", "#dbb40c", "#dbb40c"], lighting=True, show_edges=False)
    Q = QUAT.R_vector_to_vector(V_proto[2], arm_dir)
    plotter.add_lines(np.array([V_proto[0], V_proto[1],
                                V_proto[1], V_proto[2],
                                V_proto[1], V_proto[3],
                                ]), color="green")
    plotter.add_lines(np.array([V_proto[1], QUAT.rotate(Q, V_proto[2])]), color="red")
    plotter.remove_scalar_bar()

def show_proto():
    plotter = pv.Plotter()
    v, f = gen_proto()
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: f}, v)
    plotter.add_axes()
    plotter.add_mesh(grid, color="#929591", lighting=True, show_edges=True)
    plotter.show()

if __name__ == '__main__':
    show_proto()

    plotter = pv.Plotter(shape=(3, 5))
    arm_dir = normalize(np.array([2.0, 1.0, 2.0]))
    for i, t in enumerate(np.linspace(0.33, 1.0, 5)):
        tm, s = biharmonic_templating(t, arm_dir)
        plotter.subplot(0, i)
        show_deform(plotter, tm, s, arm_dir)
    arm_dir = normalize(np.array([1.0, 2.0, 2.0]))
    for i, t in enumerate(np.linspace(0.33, 1.0, 5)):
        tm, s = biharmonic_templating(t, arm_dir)
        plotter.subplot(1, i)
        show_deform(plotter, tm, s, arm_dir)
    arm_dir = normalize(np.array([2.0, 5.0, -5.0]))
    for i, t in enumerate(np.linspace(0.33, 1.0, 5)):
        tm, s = biharmonic_templating(t, arm_dir)
        plotter.subplot(2, i)
        show_deform(plotter, tm, s, arm_dir)
    plotter.show()


    plotter = pv.Plotter(shape=(3, 5))
    arm_dir = normalize(np.array([2.0, 1.0, 2.0]))
    for i, t in enumerate(np.linspace(0.33, 1.0, 5)):
        tm, s = arap_templating(t, arm_dir)
        plotter.subplot(0, i)
        show_deform(plotter, tm, s, arm_dir)
    arm_dir = normalize(np.array([1.0, 2.0, 2.0]))
    for i, t in enumerate(np.linspace(0.33, 1.0, 5)):
        tm, s = arap_templating(t, arm_dir)
        plotter.subplot(1, i)
        show_deform(plotter, tm, s, arm_dir)
    arm_dir = normalize(np.array([2.0, 5.0, -5.0]))
    for i, t in enumerate(np.linspace(0.33, 1.0, 5)):
        tm, s = arap_templating(t, arm_dir)
        plotter.subplot(2, i)
        show_deform(plotter, tm, s, arm_dir)
    plotter.show()
