import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import convolution as CONV
from tools.contouring import contour
import rainbow.math.quaternion as QUAT
import tetgen
import igl
import pyvista as pv
from tools.mesh_util import TetMesh
import polyscope as ps
# import meshio
# import data as DATA
# from tools.contouring import cluster_contour
# from tools.mesh_util import MultiTetMesh, load_tet, load_tri
# from tools.numpy_util import mk_mask
# from blocked import *
# import clusters as CLUSTERS
# import data as DATA
# import tree as TREE

ps.init()
ps.set_screenshot_extension(".png")
ps.set_window_size(1500, 1000)
ps.set_ground_plane_mode("shadow_only")
ps.set_SSAA_factor(2)
ps.set_up_dir("z_up")

def setview(mat):
    ps.set_camera_view_matrix(np.array(mat).reshape(4,4))

def add_tet(tet_mesh, id):
    return ps.register_volume_mesh(id, tet_mesh.nodes, tets=tet_mesh.tets)

WARM_GREY = (0.89, 0.808, 0.655)
DARK_GREY = (0.234, 0.325, 0.409)
PASTEL_RED = (0.512, 0.113, 0.113)
PASTEL_BLUE = (0.314, 0.511, 0.694)
PASTEL_DARK_BLUE = (0.207, 0.297, 0.523)
PASTEL_GREEN = (0.206, 0.541, 0.177)
PASTEL_YELLOW = (0.648, 0.545, 0.159)
PASTEL_PURPLE = (0.762, 0.333, 0.672)

V_proto = np.array([[ 0.0,            0.0, -1.0           ],
                    [ 0.0,            0.0,  0.0           ],
                    [ 0.5*np.sqrt(2), 0.0,  0.5*np.sqrt(2)],
                    [-0.5*np.sqrt(2), 0.0,  0.5*np.sqrt(2)]])
E_proto = np.array([[0,1], [1,2], [1,3]])
R_proto = np.array([1/3, 1/3, 1/3, 1/3])

def normalize(v):
    return v / np.linalg.norm(v, axis=-1)

def gen_proto():
    grid = CONV.conv_surf(V_proto, E_proto, R_proto, 0.05)
    v, t = contour(grid)
    tgen = tetgen.TetGen(v, t)
    nodes, elems = tgen.tetrahedralize()
    return nodes, np.int64(elems)

def biharmonic_templating(t, arm_dir):
    v, f = gen_proto()
    u = v.copy()

    handles = np.full((v.shape[0], 1), -1)
    handles[np.dot(v-t*V_proto[0], V_proto[0]) > 0] = 0
    handles[np.dot(v-t*V_proto[2], V_proto[2]) > 0] = 1
    handles[np.dot(v-t*V_proto[3], V_proto[3]) > 0] = 2

    b = np.array([[t[0] for t in [(i, handles[i]) for i in range(0, v.shape[0])] if t[1] >= 0]]).T

    ## Boundary conditions directly on deformed positions
    u_bc = np.zeros((b.shape[0], v.shape[1]))
    v_bc = np.zeros((b.shape[0], v.shape[1]))

    for bi in range(b.shape[0]):
        v_bc[bi] = v[b[bi]]
        if handles[b[bi]] == 1: # Move handle 1 over
            Q = QUAT.R_vector_to_vector(V_proto[2], arm_dir)
            u_bc[bi] = QUAT.rotate(Q, v[b[bi]])
        else: # Hold other handles still
            u_bc[bi] = v[b[bi]]

    d_bc = u_bc - v_bc
    d = igl.harmonic(v, f, b, d_bc, 2)
    u = v + d

    return TetMesh(u,f), handles.flatten()


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
            bc[i] = QUAT.rotate(Q, v[b[i]])
    vn = arap.solve(bc, v)
    return TetMesh(vn,f), s.flatten()

def show_deform(tet_mesh, s, i, j, offset):
    free_mask = np.any(s[tet_mesh.tets] == -1, axis=1)
    ps.register_volume_mesh(f"free{i}{j}", tet_mesh.nodes+offset, tet_mesh.tets[free_mask], color=DARK_GREY, interior_color=DARK_GREY, edge_width=0.5)
    ps.register_volume_mesh(f"handles{i}{j}", tet_mesh.nodes+offset, tet_mesh.tets[~free_mask], color=PASTEL_YELLOW, interior_color=PASTEL_YELLOW, edge_width=0.5)

def show_proto():
    plotter = pv.Plotter()
    v, f = gen_proto()
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: f}, v)
    plotter.add_axes()
    plotter.add_mesh(grid, color="#929591", lighting=True, show_edges=True)
    plotter.show()

if __name__ == '__main__':
    # ps.set_view_from_json('{"farClipRatio":20.0,"fov":71.3068473094254,"nearClipRatio":0.005,"projectionMode":"Orthographic","viewMat":[-0.808453798294067,-0.588559746742249,-2.91192470136536e-09,4.12101554870605,0.310961961746216,-0.427141934633255,0.849029004573822,0.4459108710289,-0.499703735113144,0.686401546001434,0.528343200683594,-6.23167085647583,0.0,0.0,0.0,1.0],"windowHeight":1122,"windowWidth":1619}')


    arm_dir0 = normalize(np.array([1,0,1]))
    arm_dir = normalize(np.array([1.0, 1, -0.7]))
    Q = QUAT.R_vector_to_vector(arm_dir0, arm_dir)

    # for i, t in enumerate(np.linspace(0.33, 1.0, 4)):
    #     for j, s in enumerate(np.linspace(0, 1, 5)[1:]):
    #         tm, handles = biharmonic_templating(t, QUAT.rotate(QUAT.slerp(QUAT.identity(), Q, s), arm_dir0))
    #         show_deform(tm, handles, i, j, np.array([2*i, 2*j, 0]))

    # v, f = gen_proto()
    # ps.register_volume_mesh(f"raw", v+np.array([3, -2, 0]), f, color=DARK_GREY, interior_color=DARK_GREY, edge_width=0.5)

    # ps.show()
    # ps.remove_all_structures()

    # for i, t in enumerate(np.linspace(0.33, 1.0, 4)):
    #     for j, s in enumerate(np.linspace(0, 1, 5)[1:]):
    #         tm, handles = arap_templating(t, QUAT.rotate(QUAT.slerp(QUAT.identity(), Q, s), arm_dir0))
    #         show_deform(tm, handles, i, j, np.array([2*i, 2*j, 0]))

    # v, f = gen_proto()
    # ps.register_volume_mesh(f"raw", v+np.array([3, -2, 0]), f, color=DARK_GREY, interior_color=DARK_GREY, edge_width=0.5)

    # ps.show()
    ps.set_view_from_json('{"farClipRatio":20.0,"fov":45.0,"nearClipRatio":0.005,"projectionMode":"Perspective","viewMat":[-0.972765386104584,-0.231795698404312,-4.5249293201266e-09,0.0427935868501663,-0.0490979142487049,0.206046760082245,0.977312028408051,0.0687019303441048,-0.226535484194756,0.950695931911469,-0.211814776062965,-3.14220476150513,0.0,0.0,0.0,1.0],"windowHeight":1000,"windowWidth":988}')


    tm, handles = biharmonic_templating(0.6, QUAT.rotate(QUAT.slerp(QUAT.identity(), Q, 1), arm_dir0))
    free_mask = np.any(handles[tm.tets] == -1, axis=1)
    ps.register_volume_mesh(f"biharm_free", tm.nodes, tm.tets[free_mask], color=DARK_GREY, interior_color=DARK_GREY, edge_width=0.5)
    ps.register_volume_mesh(f"biharm_handles", tm.nodes, tm.tets[~free_mask], color=PASTEL_YELLOW, interior_color=PASTEL_YELLOW, edge_width=0.5)

    # tm, handles = arap_templating(0.6, QUAT.rotate(QUAT.slerp(QUAT.identity(), Q, 1), arm_dir0))
    # free_mask = np.any(handles[tm.tets] == -1, axis=1)
    # ps.register_volume_mesh(f"arap_free", tm.nodes, tm.tets[free_mask], color=DARK_GREY, interior_color=DARK_GREY, edge_width=0.5)
    # ps.register_volume_mesh(f"arap_handles", tm.nodes, tm.tets[~free_mask], color=PASTEL_YELLOW, interior_color=PASTEL_YELLOW, edge_width=0.5)

    ps.show()

    print(ps.get_view_as_json())
