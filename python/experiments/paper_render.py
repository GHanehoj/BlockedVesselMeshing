import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import polyscope as ps
import meshio
import data as DATA
ps.init()
ps.set_screenshot_extension(".png")
ps.set_window_size(1500, 1000)
ps.set_ground_plane_mode("shadow_only")
ps.set_SSAA_factor(2)

def setview(mat):
    ps.set_camera_view_matrix(np.array(mat).reshape(4,4))

def add_tet(tet_mesh, id):
    return ps.register_volume_mesh(id, tet_mesh.nodes.copy(), tets=tet_mesh.tets.copy())


def show_cluster_meshes(tet, in_end, out_ends, id, offset=np.array([0,0,0])):
    ps.register_volume_mesh(f"{id}_tet", tet.nodes+offset, tet.tets, color=PASTEL_RED, interior_color=WARM_GREY, edge_width=0.5)
    msh = ps.register_surface_mesh(f"{id}_in_end", in_end.end.nodes+offset, in_end.end.tris, color=WARM_GREY, edge_width=0.5, back_face_policy="identical")
    msh.set_cull_whole_elements(True)
    for i, out_end in enumerate(out_ends):
        msh = ps.register_surface_mesh(f"{id}_out_end_{i}", out_end.end.nodes+offset, out_end.end.tris, color=WARM_GREY, edge_width=0.5, back_face_policy="identical")
        msh.set_cull_whole_elements(True)

from tools.contouring import cluster_contour
from tools.mesh_util import MultiTetMesh, load_tet, load_tri
from tools.numpy_util import mk_mask
from blocked import *
import clusters as CLUSTERS
import data as DATA
import tree as TREE
WARM_GREY = (0.89, 0.808, 0.655)
DARK_GREY = (0.234, 0.325, 0.409)
PASTEL_RED = (0.512, 0.113, 0.113)
PASTEL_BLUE = (0.314, 0.511, 0.694)
PASTEL_DARK_BLUE = (0.207, 0.297, 0.523)
PASTEL_GREEN = (0.206, 0.541, 0.177)
PASTEL_YELLOW = (0.648, 0.545, 0.159)
PASTEL_PURPLE = (0.762, 0.333, 0.672)

def load(id):
    tree_folder = f"../../data/trees/{id}"
    V, E, R = DATA.load_skeleton_data(tree_folder)
    root, _ = TREE.make_tree(V, E, R)
    # root.position += 5*(root.position - root.children[0].position)

    root_cluster = CLUSTERS.make_cluster(root.children[0])
    cluster = root_cluster.outflows[0].cluster.outflows[0].cluster.outflows[0].cluster.outflows[1].cluster.outflows[0].cluster
    parent = root_cluster.outflows[0].cluster.outflows[0].cluster.outflows[0].cluster.outflows[1].cluster

    return V, E, R, cluster, parent, root_cluster


def fig1():
    ps.set_up_dir("neg_z_up")
    setview([-0.869497001171112,-0.474276095628738,-0.13790787756443,272.307647705078,-0.00464040925726295,0.287041217088699,-0.957913458347321,445.690490722656,0.4939024746418,-0.832264542579651,-0.251783788204193,82.9010238647461,0.0,0.0,0.0,1.0])
    _, _, _, cluster, _, _ = load("orig")
    verts, tris = cluster_contour(cluster, 4)
    cluster_tet = run_tetgen(verts, tris)
    tet1 = ps.register_volume_mesh("cluster", cluster_tet.nodes, cluster_tet.tets, color=PASTEL_RED, interior_color=WARM_GREY, edge_width=0.5)
    ps.show()


def fig2():
    ps.set_up_dir("neg_z_up")
    setview([-0.869497001171112,-0.474276095628738,-0.13790787756443,272.307647705078,-0.00464040925726295,0.287041217088699,-0.957913458347321,445.690490722656,0.4939024746418,-0.832264542579651,-0.251783788204193,82.9010238647461,0.0,0.0,0.0,1.0])
    _, _, _, cluster, _, _ = load("split")
    show_cluster_meshes(*compute_cluster_meshes(cluster, 4), "cluster")
    ps.show()

def fig3():
    ps.set_up_dir("z_up")
    setview([0.797478377819061,-0.603086054325104,0.0172179844230413,-107.362449645996,0.20957238972187,0.303658992052078,0.929454505443573,-538.16357421875,-0.565768778324127,-0.737607777118683,0.368552088737488,-75.0643615722656,0.0,0.0,0.0,1.0])
    _, _, _, cluster, parent, _ = load("split")
    c_tet, c_in, c_outs = compute_cluster_meshes(cluster, 4)
    p_tet, p_in, p_outs = compute_cluster_meshes(parent, 4)
    dir = c_in.flow_data.dir
    show_cluster_meshes(c_tet, c_in, c_outs, "cluster", -2*dir)
    show_cluster_meshes(p_tet, p_in, p_outs, "parent", 2*dir)



    edge_cnt = int(np.ceil((p_outs[0].edge.segs.shape[0]+c_in.edge.segs.shape[0])/2))

    conn_mesh, conn_edge1, conn_edge2 = connector_tube(p_outs[0], c_in, edge_cnt)

    strip1 = strip(p_outs[0], conn_edge1)
    strip2 = strip(c_in, conn_edge2)


    ps.register_surface_mesh(f"conn_end_1", p_outs[0].end.nodes, p_outs[0].end.tris, color=PASTEL_BLUE, edge_width=0.5, back_face_policy="identical")
    ps.register_surface_mesh(f"conn_stri_1", strip1.nodes, strip1.tris, color=PASTEL_YELLOW, edge_width=0.5, back_face_policy="identical")
    ps.register_surface_mesh(f"conn_tube", conn_mesh.nodes, conn_mesh.tris, color=PASTEL_GREEN, edge_width=0.5, back_face_policy="identical")
    ps.register_surface_mesh(f"conn_strip_2", strip2.nodes, strip2.tris, color=PASTEL_YELLOW, edge_width=0.5, back_face_policy="identical")
    ps.register_surface_mesh(f"conn_end_2", c_in.end.nodes, c_in.end.tris, color=PASTEL_BLUE, edge_width=0.5, back_face_policy="identical")


    # conn = make_connector(p_outs[0], c_in)
    # ps.register_volume_mesh(f"conn_tet", conn.nodes, conn.tets, color=PASTEL_YELLOW, interior_color=WARM_GREY, edge_width=0.5)
    # msh = ps.register_surface_mesh(f"conn_end_1", p_outs[0].end.nodes, p_outs[0].end.tris, color=WARM_GREY, edge_width=0.5, back_face_policy="identical")
    # msh.set_cull_whole_elements(True)
    # msh = ps.register_surface_mesh(f"conn_end_2", c_in.end.nodes, c_in.end.tris, color=WARM_GREY, edge_width=0.5, back_face_policy="identical")
    # msh.set_cull_whole_elements(True)

    # c = np.mean(conn.nodes, axis=0)
    # plane = ps.add_scene_slice_plane()
    # plane.set_pose(c, (0.31, -0.1, -1))

    ps.show()



def fig4():
    ps.set_up_dir("z_up")
    setview([0.881423115730286,-0.451540619134903,0.138541981577873,-201.58349609375,0.111849829554558,0.484531372785568,0.867596864700317,-503.478576660156,-0.458884209394455,-0.749222755432129,0.477583110332489,-160.615798950195,0.0,0.0,0.0,1.0])
    _, _, _, cluster, parent, _ = load("orig")
    _, c_in, _ = compute_cluster_meshes(cluster, 4)
    _, _, p_outs = compute_cluster_meshes(parent, 4)

    edge_cnt = int(np.ceil((p_outs[0].edge.segs.shape[0]+c_in.edge.segs.shape[0])/2))

    conn_mesh, conn_edge1, conn_edge2 = connector_tube(p_outs[0], c_in, edge_cnt)

    strip1 = strip(p_outs[0], conn_edge1)
    strip2 = strip(c_in, conn_edge2)


    ps.register_surface_mesh(f"conn_end_1", p_outs[0].end.nodes, p_outs[0].end.tris, color=PASTEL_BLUE, edge_width=0.5, back_face_policy="identical")
    ps.register_surface_mesh(f"conn_stri_1", strip1.nodes, strip1.tris, color=PASTEL_YELLOW, edge_width=0.5, back_face_policy="identical")
    ps.register_surface_mesh(f"conn_tube", conn_mesh.nodes, conn_mesh.tris, color=PASTEL_GREEN, edge_width=0.5, back_face_policy="identical")
    ps.register_surface_mesh(f"conn_strip_2", strip2.nodes, strip2.tris, color=PASTEL_YELLOW, edge_width=0.5, back_face_policy="identical")
    ps.register_surface_mesh(f"conn_end_2", c_in.end.nodes, c_in.end.tris, color=PASTEL_BLUE, edge_width=0.5, back_face_policy="identical")

    ps.show()

def fig5():
    ps.set_up_dir("y_up")
    setview([-0.177589818835258,1.11758708953857e-08,-0.984103918075562,506.248168945312,-0.791556596755981,0.59416651725769,0.142843142151833,97.2533416748047,0.584721267223358,0.804342210292816,-0.105518244206905,-1342.8515625,0.0,0.0,0.0,1.0])
    ps.set_shadow_darkness(0.45)
    V, E, R, _, _, _ = load("__reg200")
    ps.register_curve_network(f"graph", V, E, color=PASTEL_BLUE)
    points = ps.register_point_cloud("nodes", V, color=PASTEL_YELLOW)
    points.add_scalar_quantity("radii", 4*R)
    points.set_point_radius_quantity("radii")
    ps.show()

def fig6():
    ps.set_up_dir("y_up")
    setview([-0.177589818835258,1.11758708953857e-08,-0.984103918075562,506.248168945312,-0.791556596755981,0.59416651725769,0.142843142151833,97.2533416748047,0.584721267223358,0.804342210292816,-0.105518244206905,-1342.8515625,0.0,0.0,0.0,1.0])
    ps.set_shadow_darkness(0.45)
    V, E, R, _, _, _ = load("__reg200")
    for i,e in enumerate(E):
        m = ps.register_curve_network(f"edge{i}", V[e], np.array([[0,1]]), color=PASTEL_BLUE)
        m.set_radius(np.mean(R[e]), relative=False)
    points = ps.register_point_cloud("nodes", V, color=PASTEL_YELLOW)
    points.add_scalar_quantity("radii", 2.1*R)
    points.set_point_radius_quantity("radii")
    ps.show()

def fig7():
    ps.set_up_dir("z_up")
    setview([-0.950029134750366,0.312157303094864,5.60005020133758e-08,351.954162597656,-0.172780841588974,-0.525848090648651,0.832846403121948,-288.299530029297,0.259979248046875,0.791229128837585,0.553506731987,-618.368286132812,0.0,0.0,0.0,1.0])
    _, _, _, cluster, parent, root = load("orig")
    cluster = root.outflows[0].cluster.outflows[0].cluster.outflows[0].cluster
    parent = root.outflows[0].cluster.outflows[0].cluster

    c_tet, c_in, _ = compute_cluster_meshes(cluster, 4)
    p_tet, _, p_outs = compute_cluster_meshes(parent, 4)
    conn = make_connector(p_outs[0], c_in)
    merged = MultiTetMesh(p_tet, [MultiTetMesh(c_tet, [], [], [], [])], [conn], [p_outs[0]], [c_in])

    edge_idxs = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])
    all_edges = merged.tets[:, edge_idxs].reshape(-1, 2)

    ps.register_curve_network(f"wireframe", merged.nodes, all_edges, color=DARK_GREY, radius=0.01)

    ps.show()


def fig8():
    ps.set_up_dir("y_up")
    setview([0.984074831008911,4.65661287307739e-10,0.177716985344887,-488.376342773438,0.126312866806984,0.703443050384521,-0.69943642616272,221.277145385742,-0.1250139772892,0.710754156112671,0.692243754863739,-543.832397460938,0.0,0.0,0.0,1.0])
    _, _, _, cluster, parent, root = load("orig")
    cluster = root.outflows[0].cluster.outflows[0].cluster.outflows[0].cluster
    parent = root.outflows[0].cluster.outflows[0].cluster

    c_tet, c_in, c_outs = compute_cluster_meshes(cluster, 4)
    p_tet, p_in, p_outs = compute_cluster_meshes(parent, 4)
    conn = make_connector(p_outs[0], c_in)
    merged = MultiTetMesh(p_tet, [MultiTetMesh(c_tet, [], [], [], [])], [conn], [p_outs[0]], [c_in])

    ps.register_volume_mesh(f"conn_tet", merged.nodes, merged.tets, color=PASTEL_RED, interior_color=WARM_GREY, edge_width=0.5)
    msh = ps.register_surface_mesh(f"p_in", p_in.end.nodes, p_in.end.tris, color=WARM_GREY, edge_width=0.5, back_face_policy="identical")
    msh.set_cull_whole_elements(True)
    for i, out in enumerate(p_outs[1:]):
        msh = ps.register_surface_mesh(f"p_out{i}", out.end.nodes, out.end.tris, color=WARM_GREY, edge_width=0.5, back_face_policy="identical")
        msh.set_cull_whole_elements(True)
    for i, out in enumerate(c_outs):
        msh = ps.register_surface_mesh(f"c_out{i}", out.end.nodes, out.end.tris, color=WARM_GREY, edge_width=0.5, back_face_policy="identical")
        msh.set_cull_whole_elements(True)

    c = np.mean(conn.nodes, axis=0)
    plane = ps.add_scene_slice_plane()
    plane.set_pose(c, (0, -1, 0))

    ps.show()


def fig9():
    ps.set_up_dir("neg_z_up")
    setview([-0.869497001171112,-0.474276095628738,-0.13790787756443,272.307647705078,-0.00464040925726295,0.287041217088699,-0.957913458347321,445.690490722656,0.4939024746418,-0.832264542579651,-0.251783788204193,82.9010238647461,0.0,0.0,0.0,1.0])
    _, _, _, cluster, _, _ = load("split")

    cluster = cluster.outflows[0].cluster
    ps.register_curve_network(f"graph", cluster.V, cluster.E, color=PASTEL_BLUE)
    for i,e in enumerate(cluster.E):
        m = ps.register_curve_network(f"edge{i}", cluster.V[e], np.array([[0,1]]), color=PASTEL_BLUE)
        # m.set_radius(np.mean(cluster.R[e]), relative=False)
    points = ps.register_point_cloud("nodes", cluster.V, color=PASTEL_YELLOW)
    # points.add_scalar_quantity("radii", 2.1*cluster.R)
    # points.set_point_radius_quantity("radii", autoscale=False)
    ps.show()


def fig10():
    ps.set_up_dir("y_up")
    setview([-0.177589818835258,1.11758708953857e-08,-0.984103918075562,506.248168945312,-0.791556596755981,0.59416651725769,0.142843142151833,97.2533416748047,0.584721267223358,0.804342210292816,-0.105518244206905,-1342.8515625,0.0,0.0,0.0,1.0])
    ps.set_shadow_darkness(0.45)
    V, E, R, _, _, _ = load("__reg200")
    for i,e in enumerate(E):
        m = ps.register_curve_network(f"edge{i}", V[e], np.array([[0,1]]), color=PASTEL_RED)
        m.set_radius(np.mean(R[e]), relative=False)
    points = ps.register_point_cloud("nodes", V, color=PASTEL_YELLOW)
    points.add_scalar_quantity("radii", 2.1*R)
    points.set_point_radius_quantity("radii")
    ps.show()

def fig11():
    ps.set_up_dir("y_up")
    setview([0.98886650800705,-2.15368345379829e-09,-0.14877861738205,-244.420639038086,-0.00454709772020578,0.999533295631409,-0.0302230324596167,-93.0576782226562,0.148709669709206,0.0305639840662479,0.988407731056213,-890.385375976562,0.0,0.0,0.0,1.0])
    _, _, _, _, _, root = load("__reg200")
    cluster = root.outflows[0].cluster.outflows[0].cluster.outflows[0].cluster.outflows[0].cluster.outflows[0].cluster.outflows[0].cluster.outflows[0].cluster.outflows[0].cluster

    verts, tris = cluster_contour(cluster, 4)
    mesh = run_tetgen(verts, tris)
    flow_data = cluster.outflows[0].data
    plane_dist, plane_proj = proj_to_plane(mesh.nodes, flow_data)
    b1, b2, _ = VEC.make_orthonormal_vectors(flow_data.dir)
    nodes2d = np.stack((np.dot(mesh.nodes, b1), np.dot(mesh.nodes, b2)), axis=-1)

    beyond_mask = plane_dist > 0
    close_mask = np.logical_and(plane_dist < 1.5*flow_data.cut_dist, np.linalg.norm(plane_proj, axis=1) < 1.5*flow_data.radius)

    cells_beyond = np.isin(mesh.tets, np.where(beyond_mask)[0])
    cells_close = np.all(np.isin(mesh.tets, np.where(close_mask)[0]), axis=1)
    cell_stat = np.sum(cells_beyond, axis=1)

    crossing_mask = np.logical_and(cell_stat == 1, cells_close)
    to_remove_mask = np.logical_and(cell_stat != 0, cells_close)

    tris = np.sort(mesh.tets[crossing_mask][~cells_beyond[crossing_mask]].reshape(-1,3), axis=1)
    crossed_idxs = mesh.tets[crossing_mask][ cells_beyond[crossing_mask]]

    normals = calc_normal_towards(mesh.nodes[tris], mesh.nodes[crossed_idxs])
    all_tet_tris = np.sort(mesh.tets[:, tet_tris], axis=2)

    tris, normals = remove_duplicate_tris(tris, normals)
    tris, normals = remove_surface_tris(all_tet_tris, tris, normals)
    tris, normals = trim_disconnected(tris, normals)

    overlap_idxs = calc_overlaps(nodes2d, tris)
    i = 0

    ps.register_volume_mesh(f"tet", mesh.nodes, mesh.tets[~to_remove_mask], color=PASTEL_RED, interior_color=WARM_GREY, edge_width=0.5)
    if len(overlap_idxs) > 0:
        trimask = mk_mask(overlap_idxs, len(tris))
        ps.register_surface_mesh(f"end", mesh.nodes, tris[~trimask], color=WARM_GREY, edge_width=0.5, back_face_policy="identical")
        ps.register_surface_mesh(f"overlap", mesh.nodes, tris[trimask], color=PASTEL_YELLOW, edge_width=0.5, back_face_policy="identical")
    else:
        ps.register_surface_mesh(f"end", mesh.nodes, tris, color=WARM_GREY, edge_width=0.5, back_face_policy="identical")
    ps.show()
    ps.remove_all_structures()

    while len(overlap_idxs) > 0:
        i += 1
        if i == 2:
            a=2
        if i > 50: raise "overlap_overflow"
        overlap_plane_dists = np.mean(plane_dist[tris[overlap_idxs]], axis=1)
        to_remove_idx = overlap_idxs[np.argmax(overlap_plane_dists)]
        to_remove_tri = tris[to_remove_idx]

        tet_idxs, ks = np.where(np.all(all_tet_tris == to_remove_tri[None, None, :], axis=2))
        assert(len(tet_idxs) == 2)

        tet_top_dot_normal = np.dot(mesh.nodes[mesh.tets[tet_idxs, tet_tops[ks]]]-mesh.nodes[to_remove_tri[0]], normals[to_remove_idx])
        lower_tet_idx = np.argmin(tet_top_dot_normal)

        to_remove_mask[tet_idxs[lower_tet_idx]] = True

        new_tris = np.sort(mesh.tets[tet_idxs[lower_tet_idx], tet_other_tris[ks[lower_tet_idx]]], axis=1)
        new_normals = calc_normal_towards(mesh.nodes[new_tris], np.mean(mesh.nodes[to_remove_tri], axis=0))

        tris = np.concatenate((np.delete(tris, [to_remove_idx], axis=0), new_tris))
        normals = np.concatenate((np.delete(normals, [to_remove_idx], axis=0), new_normals))

        tris, normals = remove_duplicate_tris(tris, normals)
        tris, normals = remove_surface_tris(all_tet_tris, tris, normals)
        tris, normals = trim_disconnected(tris, normals)

        overlap_idxs = calc_overlaps(nodes2d, tris)

        ps.register_volume_mesh(f"tet", mesh.nodes, mesh.tets[~to_remove_mask], color=PASTEL_RED, interior_color=WARM_GREY, edge_width=0.5)
        if len(overlap_idxs) > 0:
            trimask = mk_mask(overlap_idxs, len(tris))
            ps.register_surface_mesh(f"end", mesh.nodes, tris[~trimask], color=WARM_GREY, edge_width=0.5, back_face_policy="identical")
            ps.register_surface_mesh(f"overlap", mesh.nodes, tris[trimask], color=PASTEL_YELLOW, edge_width=0.5, back_face_policy="identical")
        else:
            ps.register_surface_mesh(f"end", mesh.nodes, tris, color=WARM_GREY, edge_width=0.5, back_face_policy="identical")

        ps.register_volume_mesh(f"removed", mesh.nodes, mesh.tets[tet_idxs[lower_tet_idx]].reshape(1,4), color=PASTEL_GREEN, transparency=0.65, interior_color=WARM_GREY, material="flat")
        ps.show()
        ps.remove_all_structures()


    ps.register_volume_mesh(f"{id}_tet", mesh.nodes, mesh.tets[~to_remove_mask], color=PASTEL_RED, interior_color=WARM_GREY, edge_width=0.5)
    ps.register_surface_mesh(f"{id}_out_end", mesh.nodes, tris, color=WARM_GREY, edge_width=0.5, back_face_policy="identical")
    ps.show()

def fig12():
    ps.set_up_dir("y_up")
    setview([-0.177589818835258,1.11758708953857e-08,-0.984103918075562,506.248168945312,-0.791556596755981,0.59416651725769,0.142843142151833,97.2533416748047,0.584721267223358,0.804342210292816,-0.105518244206905,-1342.8515625,0.0,0.0,0.0,1.0])
    ps.set_shadow_darkness(0.45)
    ps.set_SSAA_factor(1)
    tet = load_tet("../../data/meshes/full_4.mesh")
    print("loaded")
    ps.register_volume_mesh("cluster", tet.nodes, tet.tets[:100000], material="flat",color=PASTEL_RED, interior_color=WARM_GREY, edge_width=0.5)
    print("added")
    ps.show()

def fig13():
    ps.set_up_dir("y_up")
    ps.set_shadow_darkness(0.45)
    tri = load_tri("../../data/meshes/full_4_surface.mesh")
    print("loaded")
    ps.register_surface_mesh("cluster", tri.nodes, tri.tris[:2000000], color=PASTEL_RED, edge_width=0.5, back_face_policy="identical")
    print("added")
    setview([-0.177589818835258,1.11758708953857e-08,-0.984103918075562,241.179550170898,-0.791556596755981,0.59416651725769,0.142843142151833,6.04909515380859,0.584721267223358,0.804342210292816,-0.105518244206905,-647.394287109375,0.0,0.0,0.0,1.0])
    ps.screenshot(filename="img1.png", transparent_bg=False)
    setview([-0.966111719608307,4.86033968627453e-09,0.258117377758026,201.167755126953,0.0315577276051044,0.992499053478241,0.118117302656174,-299.370147705078,-0.256182014942169,0.12226128578186,-0.958864986896515,166.086456298828,0.0,0.0,0.0,1.0])
    ps.screenshot(filename="img2.png", transparent_bg=False)
    setview([0.870456039905548,9.31322574615479e-09,-0.492244213819504,-222.858261108398,-0.163492113351822,0.943231403827667,-0.289109975099564,-71.0966720581055,0.464300334453583,0.332136392593384,0.821041464805603,-935.004760742188,0.0,0.0,0.0,1.0])
    ps.screenshot(filename="img3.png", transparent_bg=False)
    setview([0.0782720074057579,1.48429535329342e-09,-0.996930837631226,766.173522949219,-0.754002392292023,0.654201865196228,-0.0591984577476978,249.424880981445,0.652197062969208,0.756321430206299,0.051205288618803,-391.732513427734,0.0,0.0,0.0,1.0])
    ps.screenshot(filename="out.png", transparent_bg=False)
    ps.show()


def load_feq(file):
    msh = meshio.read(file)
    verts = msh.points
    quads = msh.cells_dict["quad"]
    tris = quads[:, [[0,1,2],[0,2,3]]].reshape(-1, 3)
    return TriMesh(verts, tris)

def fig14():
    ps.set_up_dir("y_up")
    ps.set_shadow_darkness(0.45)
    tri = load_feq("../../data/meshes/feq_full.obj")
    print("loaded")
    ps.register_surface_mesh("cluster", tri.nodes, tri.tris, color=PASTEL_RED, edge_width=0.5, back_face_policy="identical")
    print("added")
    setview([-0.177589818835258,1.11758708953857e-08,-0.984103918075562,241.179550170898,-0.791556596755981,0.59416651725769,0.142843142151833,6.04909515380859,0.584721267223358,0.804342210292816,-0.105518244206905,-647.394287109375,0.0,0.0,0.0,1.0])
    ps.show()



fig13()

print(ps.get_view_as_json())
