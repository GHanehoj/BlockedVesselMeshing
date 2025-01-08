import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import tetgen
import rainbow.math.quaternion as QUAT
import rainbow.math.vector3 as VEC
from tools.mesh_util import SegMesh, TriMesh, TetMesh, merge_tri_meshes, merge_tet_meshes, clean_tet_mesh
from tools.contouring import branch_contour
from branch import mk_branch
from tree import TreeConf
from collections import namedtuple
EndSlice = namedtuple("EndSlice", ["end", "edge", "point", "dir", "radius"])

def normalize(v):
    return v / np.linalg.norm(v, axis=-1)

def lerp(v0, v1, t):
    assert(t >= 0 and t <= 1)
    return v0 + t*(v1-v0)

def extract_slice(nodes, cells, point, dir, radius):
    mask = np.dot(nodes-point, dir) > 0
    beyond_idxs = np.where(mask)[0]

    cells_mask = np.any(cells[:,:,None] == beyond_idxs, axis=2)

    single_pruned_cells = (np.sum(cells_mask, axis=1) == 1)
    tris = cells[single_pruned_cells][~cells_mask[single_pruned_cells]].reshape(-1, 3)

    duplicates = np.triu(np.all(np.any(tris[:, None, :, None] == tris[None, :, None, :], axis=3), axis=2), k=1)
    duplicate_mask = np.logical_or(np.any(duplicates, axis=0), np.any(duplicates, axis=1))
    tris = tris[~duplicate_mask]

    close_mask = np.max(np.linalg.norm(nodes[tris]-point, axis=2), axis=1) < 1.3*radius
    tris = tris[close_mask]

    return TriMesh(nodes, tris)

def remove_tip(nodes, cells, point, dir):
    mask = np.dot(nodes-point, dir) > 0
    beyond_idxs = np.where(mask)[0]

    cells_mask = np.any(cells[:,:,None] == beyond_idxs, axis=2)

    single_pruned_cells = (np.sum(cells_mask, axis=1) == 0)

    cells_pruned = cells[single_pruned_cells]

    return cells_pruned

def arm_info(branch, arm):
    pos = branch.positions
    diff = pos[arm]-pos[1]
    dist = np.linalg.norm(diff)
    dir = diff/dist
    point = pos[1] + 2.214*branch.radii[1]*dir
    radius = lerp(branch.radii[1], branch.radii[arm], 2.214*branch.radii[1]/dist)
    return point, dir, radius

def compute_branch_meshes(branch):
    verts, tris = branch_contour(branch)
    tgen = tetgen.TetGen(verts, tris)
    nodes, elems = tgen.tetrahedralize()

    ends = [None, None, None, None]
    for i in branch.arms:
        point, dir, radius = arm_info(branch, i)
        end = extract_slice(nodes, elems, point, dir, radius)
        edge = surface_edge(end)
        ends[i] = EndSlice(end, edge, point, dir, radius)
        elems = remove_tip(nodes, elems, point, dir)

    return TetMesh(nodes, elems), ends

def connector_tube(end1, end2, ang_res):
    v_p = lerp(end1.point, end2.point, 0.02)
    v_c = lerp(end1.point, end2.point, 0.98)
    r_p = lerp(end1.radius, end2.radius, 0.02)
    r_c = lerp(end1.radius, end2.radius, 0.98)
    lin_res = int(np.ceil((np.linalg.norm(v_p-v_c)*ang_res)/((r_p+r_c)*np.pi)))
    return tube(v_p, v_c, r_p*1.02, r_c*1.02, ang_res, lin_res)

def tube(p0, p1, r0, r1, ang_res, lin_res):
    dp = np.linalg.norm(p1-p0)
    dir = (p1-p0)/dp

    theta = np.linspace(0, 2*np.pi, ang_res, endpoint=False)
    phi = np.linspace(0, 1, lin_res+1)
    r = r0 + phi * (r1 - r0)
    x = r[:, None] * np.cos(theta[None, :])
    y = r[:, None] * np.sin(theta[None, :])
    z = np.tile(phi*dp, [ang_res, 1]).T

    Q = QUAT.R_vector_to_vector(VEC.k(), dir)
    nodes = np.dstack((x,y,z))
    nodes = nodes.reshape(-1, 3)
    nodes = QUAT.rotate_array(Q, nodes) + p0

    i = np.arange(ang_res)
    ip1 = np.mod(i+1, ang_res)
    j = np.arange(lin_res)
    tri_strip = np.vstack((np.array([i, ip1, i+ang_res]).T, np.array([ip1, i+ang_res, ip1+ang_res]).T))
    tris = tri_strip[None, :, :] + ang_res*j[:, None, None]
    tris = tris.reshape(-1, 3)

    edge1 = np.array([i, ip1]).T
    edge2 = edge1 + ang_res*lin_res

    return TriMesh(nodes, tris), SegMesh(nodes, edge1), SegMesh(nodes, edge2)

def rot_angle(p):
    return np.arctan2(np.dot(np.cross(p, VEC.i()), VEC.j()), np.dot(p, VEC.i()))
def ang_mod(angs):
    return np.where(angs < -np.pi, angs + 2*np.pi, angs)
def sort_ring(points):
    permutation = np.argsort(rot_angle(points))
    return points[permutation]
def sort_ring2(points, lines):
    n = lines.shape[0]
    permutation = np.empty((n), dtype=np.int64)
    i0 = 0
    permutation[0] = lines[i0, 0]
    while True:
        matches = lines[np.any(lines==permutation[0], axis=1)]
        next_nums = matches[matches != permutation[0]]
        angles = rot_angle(points[next_nums])-rot_angle(points[permutation[0]])
        if np.sign(angles[0]) != np.sign(angles[1]):
            currentLine = matches[np.argsort(angles)[0]]
            break
        else:
            i0 += 1
            permutation[0] = lines[i0, 0]

    for i in range(1,n):
        matches = lines[np.any(lines==permutation[i-1], axis=1)]
        nextLine = matches[np.any(matches != currentLine, axis=1)]
        next_num = nextLine[nextLine != permutation[i-1]]
        permutation[i] = next_num
        currentLine = nextLine

    return points[permutation]

def strip(end, cyl_edge):
    Q = QUAT.R_vector_to_vector(end.dir, VEC.j())
    def transform(p):     return QUAT.rotate_array(Q, p-end.point)
    def transform_inv(p): return QUAT.rotate_array(QUAT.conjugate(Q), p)+end.point

    cyl_points = sort_ring(transform(np.unique(cyl_edge.nodes[cyl_edge.segs].reshape(-1, 3), axis=0)))
    end_points = sort_ring2(transform(end.edge.nodes), end.edge.segs)

    tris = []
    i = 0
    j = 0
    angles = ang_mod(rot_angle(cyl_points)-rot_angle(end_points[i]))
    j0 = np.argmin(np.where(angles > 0, angles, np.inf))
    last_ang = angles[j0]+rot_angle(end_points[i])

    def end_p(i): return end_points[i % end_points.shape[0]]
    def cyl_p(j): return cyl_points[(j+j0) % cyl_points.shape[0]]
    def end_i(i): return i % end_points.shape[0]
    def cyl_i(j): return end_points.shape[0] + ((j+j0) % cyl_points.shape[0])

    while i < end_points.shape[0] and j < cyl_points.shape[0]:
        ang0 = ang_mod(rot_angle(cyl_p(j+1))-last_ang)
        ang1 = ang_mod(rot_angle(end_p(i+1))-last_ang)

        if ang0 > ang1:
            tris.append([cyl_i(j), end_i(i), end_i(i+1)])
            last_ang = rot_angle(end_p(i+1))
            i += 1
        else:
            tris.append([cyl_i(j), end_i(i), cyl_i(j+1)])
            last_ang = rot_angle(cyl_p(j+1))
            j += 1

    while i < end_points.shape[0]:
        tris.append([cyl_i(j), end_i(i), end_i(i+1)])
        i += 1
    while j < cyl_points.shape[0]:
        tris.append([cyl_i(j), end_i(i), cyl_i(j+1)])
        j += 1

    nodes = np.concatenate((end_points, cyl_points), axis=0)
    nodes = transform_inv(nodes)
    tris = np.array(tris)

    return TriMesh(nodes, tris)

def surface_edge(tri_mesh):
    edges = []
    edge_idxs = [[0,1], [1,2], [2,0]]

    all_edges = np.sort(tri_mesh.tris[:, edge_idxs], axis=2).reshape(-1, 2)
    ns = np.sum(np.all(all_edges[:, None, :] == all_edges[None, :, :], axis=2), axis=1)
    edges = np.unique(all_edges.reshape(-1, 3, 2)[np.where(ns.reshape(-1, 3) != 2)], axis=0)

    return SegMesh(tri_mesh.nodes, edges)

def gen(node, parent_end = None, depth = 0):
    if len(node.children) != 2: return TetMesh(np.zeros((0,3)),np.zeros((0,4), dtype=np.int64))
    if (depth > 3): return TetMesh(np.zeros((0,3)),np.zeros((0,4), dtype=np.int64))

    res = []
    branch = mk_branch(node, TreeConf("regularized", True))

    branch_tet, branch_ends = compute_branch_meshes(branch)
    res.append(clean_tet_mesh(branch_tet))

    if parent_end:
        edge_cnt = int(np.ceil(0.8*(branch_ends[0].edge.segs.shape[0]+parent_end.edge.segs.shape[0])/2))

        conn_mesh, conn_edge1, conn_edge2 = connector_tube(parent_end, branch_ends[0], edge_cnt)
        
        strip1 = strip(parent_end, conn_edge1)
        strip2 = strip(branch_ends[0], conn_edge2)

        connector_tri = merge_tri_meshes([branch_ends[0].end, strip1, conn_mesh, strip2, parent_end.end])

        tet = tetgen.TetGen(connector_tri.nodes, connector_tri.tris)
        nodes, elems = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
        connector_tet = TetMesh(nodes, elems)

        res.append(connector_tet)


    child_ends = [branch_ends[2], branch_ends[3]] if node.children[0].radius > node.children[1].radius else [branch_ends[3], branch_ends[2]]

    res.append(gen(node.children[0], child_ends[0], depth+1))
    res.append(gen(node.children[1], child_ends[1], depth+1))

    return merge_tet_meshes(res)

import data as DATA
import tree as TREE
from tools.pyvista_plotting import show_tet_mesh
from tools.mesh_util import tet_mesh_size

tree_folder = f"../data/trees/regularized"
V, E, R = DATA.load_skeleton_data(tree_folder)
root, _ = TREE.make_tree(V, E, R)

merged = gen(root.children[0])


show_tet_mesh(merged)
print(tet_mesh_size(merged), "MB")
