import numpy as np
import tetgen
import rainbow.math.quaternion as QUAT
import rainbow.math.intersection as INSCT
import rainbow.math.vector3 as VEC
from tools.mesh_util import SegMesh, TriMesh, TetMesh
from clusters import FlowData
from tools.contouring import cluster_contour
from tools.numpy_util import ang_mod, angle_in_plane, angle_xy, lerp
from collections import namedtuple
EndSlice = namedtuple("EndSlice", ["end", "edge", "flow_data"])

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


def process_end(mesh: TetMesh, flow_data: FlowData):
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
    while len(overlap_idxs) > 0:
        i += 1
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

    mesh.tets = mesh.tets[~to_remove_mask]

    tri_nodes = np.unique(tris)
    mesh.nodes[tri_nodes] -= plane_dist[tri_nodes, None]*flow_data.dir[None, :]

    end = TriMesh(mesh.nodes, tris)
    edge = end.edge()

    edge.clean()
    end.clean()
    mesh.clean()
    return EndSlice(end, edge, flow_data)

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
    _, inv, cnts = np.unique(tris, axis=0, return_inverse=True, return_counts=True)
    duplicate_mask = cnts[inv] == 2
    return tris[~duplicate_mask], normals[~duplicate_mask]

def calc_overlaps(nodes2d, tris):
    tris_flat = nodes2d[tris]
    overlaps = np.triu(INSCT.tri_intersect2(tris_flat, tris_flat), k=1)
    return np.unique(np.where(overlaps))

def calc_normal_towards(tri, p):
    vec = p-tri[...,0,:]
    normal = np.cross(tri[...,1,:]-tri[...,0,:], tri[...,2,:]-tri[...,0,:])
    normal /= np.linalg.norm(normal, axis=-1)[...,None]
    normal[np.einsum("td,td->t", normal, vec) < 0] *= -1

    return normal

def compute_cluster_meshes(cluster, res=4):
    verts, tris = cluster_contour(cluster, res)
    tgen = tetgen.TetGen(verts, tris)
    nodes, tets = tgen.tetrahedralize()
    mesh = TetMesh(nodes, tets)

    out_ends = [None]*len(cluster.outflows)
    for i, outflow in enumerate(cluster.outflows):
        out_ends[i] = process_end(mesh, outflow.data)
    in_end = process_end(mesh, cluster.in_data)

    return mesh, in_end, out_ends

def connector_tube(end1, end2, ang_res):
    v_p = lerp(end1.flow_data.point, end2.flow_data.point, 0.1)
    v_c = lerp(end1.flow_data.point, end2.flow_data.point, 0.9)
    r_p = lerp(end1.flow_data.radius, end2.flow_data.radius, 0.1)
    r_c = lerp(end1.flow_data.radius, end2.flow_data.radius, 0.9)
    lin_res = int(np.ceil((np.linalg.norm(v_p-v_c)*ang_res)/((r_p+r_c)*np.pi)/4))
    return tube(v_p, v_c, r_p, r_c, ang_res, lin_res)

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
    Q = QUAT.R_vector_to_vector(end.flow_data.dir, VEC.k())
    def transform(p):     return QUAT.rotate(Q, p-end.flow_data.point)
    def transform_inv(p): return QUAT.rotate(QUAT.conjugate(Q), p)+end.flow_data.point

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
