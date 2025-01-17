import numpy as np
import tetgen
import rainbow.math.quaternion as QUAT
import rainbow.math.intersection as INSCT
import rainbow.math.vector3 as VEC
from tools.mesh_util import SegMesh, TriMesh, TetMesh
from tools.contouring import cluster_contour
from tools.numpy_util import ang_mod, angle_in_plane, angle_xy, lerp, mk_mask
from tools.smoothing import laplacian_smoothing
from collections import namedtuple
EndSlice = namedtuple("EndSlice", ["end", "edge", "point", "dir", "radius"])

tet_tris = np.array([[0,1,2], [1,2,3], [0,2,3], [0,1,3]])
tet_other_tris = np.array([
    [[1,2,3], [0,2,3], [0,1,3]],
    [[0,1,2], [0,2,3], [0,1,3]],
    [[0,1,2], [1,2,3], [0,1,3]],
    [[0,1,2], [1,2,3], [0,2,3]],
])
tet_tops = np.array([3,0,1,2])
tri_edges = np.array([[0,1], [1,2], [2,0]])

def proj_to_plane(p, flow_data):
    p0 = p-flow_data.point
    dist = np.einsum("...d,d->...", p0, flow_data.dir)
    proj = p0 - dist[..., :, None]*flow_data.dir[..., None, :]
    return dist, proj

def extract_slice(nodes, cells, flow_data):
    plane_dist, plane_proj = proj_to_plane(nodes, flow_data)
    b1, b2, _ = VEC.make_orthonormal_vectors(flow_data.dir)
    nodes_mask = plane_dist > 0
    cells_mask = np.any(cells[:,:,None] == np.where(nodes_mask)[0], axis=2)

    single_pruned_cells = (np.sum(cells_mask, axis=1) == 1)
    tris = np.sort(cells[single_pruned_cells][~cells_mask[single_pruned_cells]].reshape(-1, 3), axis=1)
    
    beyond_idx = cells[single_pruned_cells][cells_mask[single_pruned_cells]]
    normals = calc_normal_towards(nodes[tris], nodes[beyond_idx])

    tris, normals = remove_duplicate_tris(tris, normals)

    close_mask = np.max(np.linalg.norm(plane_proj[tris], axis=2), axis=1) < 1.5*flow_data.radius
    tris = tris[close_mask]
    normals = normals[close_mask]

    all_tet_tris = np.sort(cells[:, tet_tris], axis=2)

    tris, normals = remove_surface_tris(all_tet_tris, tris, normals)
    tris, normals = trim_disconnected(tris, normals)

    removed_tets = []


    overlap_idxs = calc_overlaps(nodes, b1, b2, tris)
    i = 0
    while len(overlap_idxs) > 0:
        i += 1
        if i > 50: raise "overlap_overflow"
        overlap_plane_dists = np.mean(plane_dist[tris[overlap_idxs]], axis=1)
        to_remove_idx = overlap_idxs[np.argmax(overlap_plane_dists)]
        to_remove_tri = tris[to_remove_idx]

        tet_idxs, ks = np.where(np.all(all_tet_tris == to_remove_tri[None, None, :], axis=2))
        assert(len(tet_idxs) == 2)

        tet_top_dot_normal = np.dot(nodes[cells[tet_idxs, tet_tops[ks]]]-nodes[to_remove_tri[0]], normals[to_remove_idx])
        lower_tet_idx = np.argmin(tet_top_dot_normal)

        removed_tets += [tet_idxs[lower_tet_idx]]

        new_tris = np.sort(cells[tet_idxs[lower_tet_idx], tet_other_tris[ks[lower_tet_idx]]], axis=1)
        new_normals = calc_normal_towards(nodes[new_tris], np.mean(nodes[to_remove_tri], axis=0))

        tris = np.concatenate((np.delete(tris, [to_remove_idx], axis=0), new_tris))
        normals = np.concatenate((np.delete(normals, [to_remove_idx], axis=0), new_normals))

        tris, normals = remove_duplicate_tris(tris, normals)
        tris, normals = remove_surface_tris(all_tet_tris, tris, normals)
        tris, normals = trim_disconnected(tris, normals)

        overlap_idxs = calc_overlaps(nodes, b1, b2, tris)


    return TriMesh(nodes, tris), removed_tets
def find_connectivity(tris):
    all_tri_edges = np.sort(tris[:, tri_edges], axis=2)
    group_map = np.arange(len(tris))
    conn = np.stack(np.where(np.triu(np.any(np.all(all_tri_edges[:, None, :, None, :] == all_tri_edges[None, :, None, :, :], axis=4), axis=(2,3)),k=1))).T
    for ai, bi in conn:
        mai = group_map[ai]
        mbi = group_map[bi]
        if mbi == mai: continue
        group_map[group_map == mai] = mbi
    return group_map, *np.unique(group_map, return_counts=True)

def trim_disconnected(tris, normals):
    group_map, groups, group_szs = find_connectivity(tris)
    if len(groups) == 1: return tris, normals
    
    biggest_group_idx = np.argmax(group_szs)
    biggest_group = groups[biggest_group_idx]
    group_mask = group_map == biggest_group

    return tris[group_mask], normals[group_mask]

def remove_surface_tris(all_tet_tris, tris, normals):
    eqs = np.all(all_tet_tris[:, :, None, :] == tris[None, None, :, :], axis=3)
    surface_mask = np.sum(eqs, axis=(0,1)) == 1
    return tris[~surface_mask], normals[~surface_mask]

def remove_duplicate_tris(tris, normals):
    duplicates = np.triu(np.all(tris[:, None, :] == tris[None, :, :], axis=2), k=1)
    duplicate_mask = np.logical_or(np.any(duplicates, axis=0), np.any(duplicates, axis=1))
    return tris[~duplicate_mask], normals[~duplicate_mask]

def calc_overlaps(nodes, b1, b2, tris):
    nodes2d = np.stack((np.dot(nodes, b1), np.dot(nodes, b2)), axis=-1)
    tris_flat = nodes2d[tris]
    overlaps = np.triu(INSCT.tri_intersect2(tris_flat, tris_flat), k=1)
    return np.unique(np.where(overlaps))

def calc_normal_towards(tri, p):
    vec = p-tri[...,0,:]
    normal = np.cross(tri[...,1,:]-tri[...,0,:], tri[...,2,:]-tri[...,0,:])
    normal /= np.linalg.norm(normal, axis=-1)[...,None]
    normal[np.einsum("td,td->t", normal, vec) < 0] *= -1

    return normal

def remove_tip(nodes, cells, point, dir, radius, cut_dist):
    nodes0 = nodes-point
    dist_to_plane = np.dot(nodes0, dir)
    point_in_plane = nodes0-dist_to_plane[:, None]*dir[None, :]
    mask = np.logical_and(np.logical_and(dist_to_plane > 0.0001, dist_to_plane < 1.5*cut_dist), np.linalg.norm(point_in_plane, axis=1) < 1.5*radius)
    beyond_idxs = np.where(mask)[0]

    cells_mask = np.any(cells[:,:,None] == beyond_idxs, axis=2)

    single_pruned_cells = (np.sum(cells_mask, axis=1) == 0)

    cells_pruned = cells[single_pruned_cells]

    return cells_pruned

def compute_cluster_meshes(cluster, res=3):
    verts, tris = cluster_contour(cluster, res)
    tgen = tetgen.TetGen(verts, tris)
    nodes, tets = tgen.tetrahedralize()
    mesh = TetMesh(nodes, tets)

    out_ends = [None]*len(cluster.outflows)
    for i, outflow in enumerate(cluster.outflows):
        out_ends[i] = process_end(mesh, outflow.data)
    in_end = process_end(mesh, cluster.in_data)

    return mesh, in_end, out_ends

def process_end(mesh, flow_data):
    end, removed_tets = extract_slice(mesh.nodes, mesh.tets, flow_data)
    edge = end.edge()
    neighbours = end.calc_neighbours()
    # def show():
    #     import pyvista as pv
    #     from tools.pyvista_plotting import add_tri_mesh, add_tet_mesh
    #     plt = pv.Plotter()
    #     add_tet_mesh(plt, mesh)
    #     add_tri_mesh(plt, end)
    #     plt.show()
    Q = QUAT.R_vector_to_vector(flow_data.dir, VEC.k())
    def transform(p):     return QUAT.rotate(Q, p-flow_data.point)
    def transform_inv(p): return QUAT.rotate(QUAT.conjugate(Q), p)+flow_data.point
    
    _, sorted_idxs = sort_ring_morph(transform(mesh.nodes), edge.segs)

    n = len(sorted_idxs)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    theta0 = angle_xy(transform(mesh.nodes[sorted_idxs[0]]))
    x = flow_data.radius * np.cos(theta-theta0)
    y = flow_data.radius * np.sin(theta-theta0)
    z = np.zeros(n)

    circle_nodes = np.dstack((x,y,z))
    circle_nodes = circle_nodes.reshape(-1, 3)
    circle_nodes = transform_inv(circle_nodes)

    mesh.nodes[sorted_idxs] = circle_nodes

    laplacian_smoothing(mesh.nodes, neighbours, np.logical_or(~mk_mask(np.unique(end.tris), len(mesh.nodes)), mk_mask(sorted_idxs, len(mesh.nodes))), lr=0.2, iter=50)

    mesh.tets = mesh.tets[~mk_mask(np.array(removed_tets), len(mesh.tets))]
    mesh.tets = remove_tip(mesh.nodes, mesh.tets, flow_data.point, flow_data.dir, flow_data.radius, flow_data.cut_dist)
    return EndSlice(end, edge, flow_data.point, flow_data.dir, flow_data.radius)


def connector_tube(end1, end2, ang_res):
    v_p = lerp(end1.point, end2.point, 0.1)
    v_c = lerp(end1.point, end2.point, 0.9)
    r_p = lerp(end1.radius, end2.radius, 0.1)
    r_c = lerp(end1.radius, end2.radius, 0.9)
    lin_res = int(np.ceil((np.linalg.norm(v_p-v_c)*ang_res)/((r_p+r_c)*np.pi)/4))
    return tube(v_p, v_c, r_p, r_c, ang_res, lin_res)

def connector_tube2(end1, end2, ang_res):
    v_p = lerp(end1.point, end2.point, 0.1)
    v_c = lerp(end1.point, end2.point, 0.9)
    r_p = lerp(end1.radius, end2.radius, 0.1)
    r_c = lerp(end1.radius, end2.radius, 0.9)
    lin_res = int(np.ceil((np.linalg.norm(v_p-v_c)*ang_res)/((r_p+r_c)*np.pi)/4))
    return tube(end1.point, end2.point, end1.radius, end2.radius, ang_res, lin_res)

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
    nodes = QUAT.rotate(Q, nodes) + p0

    i = np.arange(ang_res)
    ip1 = np.mod(i+1, ang_res)
    j = np.arange(lin_res)
    tri_strip = np.vstack((np.array([i, ip1, i+ang_res]).T, np.array([ip1, i+ang_res, ip1+ang_res]).T))
    tris = tri_strip[None, :, :] + ang_res*j[:, None, None]
    tris = tris.reshape(-1, 3)

    edge1 = np.array([i, ip1]).T
    edge2 = edge1 + ang_res*lin_res

    return TriMesh(nodes, tris), SegMesh(nodes, edge1), SegMesh(nodes, edge2)

def sort_ring_geom(points):
    permutation = np.argsort(-angle_xy(points))
    return points[permutation]
def sort_ring_morph(points, lines):
    n = lines.shape[0]
    permutation = np.empty((n), dtype=np.int64)
    ang_tot = 0.0
    currentLine = lines[0]
    permutation[0] = currentLine[0]

    for i in range(1,n):
        matches = lines[np.any(lines==permutation[i-1], axis=1)]
        nextLine = matches[np.any(matches != currentLine, axis=1)][0]
        next_num = nextLine[nextLine != permutation[i-1]][0]
        ang_tot += angle_in_plane(points[permutation[i-1]], points[next_num], VEC.k())
        permutation[i] = next_num
        currentLine = nextLine

    if (ang_tot < 0):
        permutation = permutation[::-1]
    return points[permutation], permutation

def strip(end, cyl_edge):
    Q = QUAT.R_vector_to_vector(end.dir, VEC.k())
    def transform(p):     return QUAT.rotate(Q, p-end.point)
    def transform_inv(p): return QUAT.rotate(QUAT.conjugate(Q), p)+end.point

    cyl_points = sort_ring_geom(transform(np.unique(cyl_edge.nodes[cyl_edge.segs].reshape(-1, 3), axis=0)))
    end_points, _ = sort_ring_morph(transform(end.edge.nodes), end.edge.segs)

    tris = []
    i = 0
    j = 0
    angles = ang_mod(angle_xy(cyl_points)-angle_xy(end_points[i]))
    j0 = np.argmin(np.where(angles > 0, angles, np.inf))
    last_ang = angle_xy(cyl_points[j0])
    # def show():
    #     import pyvista as pv
    #     plotter = pv.Plotter()
    #     plotter.add_points(cyl_points)
    #     plotter.add_points(end_points)
    #     plotter.add_points(cyl_p(j), color="r", point_size=5)
    #     plotter.add_points(cyl_p(j+1), color="r", point_size=3)
    #     plotter.add_points(end_p(i), color="g", point_size=5)
    #     plotter.add_points(end_p(i+1), color="g", point_size=3)
    #     nodes = np.concatenate((end_points, cyl_points), axis=0)
    #     if len(tris) > 0:
    #         tris222 = np.array(tris)
    #         tris222 = np.hstack((np.full((tris222.shape[0], 1), 3), tris222))
    #         plotter.add_mesh(pv.PolyData(nodes, tris222))

        # plotter.show()
    def end_p(i): return end_points[i % end_points.shape[0]]
    def cyl_p(j): return cyl_points[(j+j0) % cyl_points.shape[0]]
    def end_i(i): return i % end_points.shape[0]
    def cyl_i(j): return end_points.shape[0] + ((j+j0) % cyl_points.shape[0])

    while i < end_points.shape[0] and j < cyl_points.shape[0]:
        ang0 = ang_mod(angle_xy(cyl_p(j+1))-last_ang)
        ang1 = ang_mod(angle_xy(end_p(i+1))-last_ang)

        if ang0 < ang1:
            tris.append([cyl_i(j), end_i(i), end_i(i+1)])
            last_ang = angle_xy(end_p(i+1))
            i += 1
        else:
            tris.append([cyl_i(j), end_i(i), cyl_i(j+1)])
            last_ang = angle_xy(cyl_p(j+1))
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
