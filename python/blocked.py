import numpy as np
import rainbow.math.quaternion as QUAT
import rainbow.math.intersection as INSCT
import rainbow.math.vector3 as VEC
from tools.mesh_util import SegMesh, TriMesh, TetMesh
from clusters import FlowData
from tools.contouring import contour
import convolution as CONV
from tools.numpy_util import angle_in_plane, angle_xy, lerp
from tools.mesh_util import merge_tri_meshes, flatness
from collections import namedtuple
import subprocess
import meshio
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
    close_mask = np.logical_and(plane_dist < 4*flow_data.radius, np.linalg.norm(plane_proj, axis=1) < 1.6*flow_data.radius)

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

    overlap_idxs = calc_overlaps(nodes2d, tris, flow_data.radius)
    i = 0
    while len(overlap_idxs) > 0:
        i += 1
        if i > 50: raise Exception("overlap_overflow")
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

        overlap_idxs = calc_overlaps(nodes2d, tris, flow_data.radius)

    # Remove boundary sliver triangles
    edge_segs = TriMesh(mesh.nodes, tris).edge().segs
    all_tri_edges = np.sort(tris[:, tri_edges], axis=2)
    border_edges = np.any(np.all(all_tri_edges[:,None,:,:] == edge_segs[None,:,None,:], axis=3), axis=1)
    flap_mask = np.sum(border_edges, axis=1) == 2

    in_nodes = all_tri_edges[flap_mask][np.arange(len(border_edges[flap_mask])),np.argmin(border_edges[flap_mask], axis=1)]
    out_nodes = np.array([np.setdiff1d(np.unique(tri_edges), in_node)[0] for tri_edges, in_node in zip(all_tri_edges[flap_mask], in_nodes)], dtype=np.int64)
    flap_dir = mesh.nodes[out_nodes] - mesh.nodes[in_nodes[:,0]]
    flap_down_mask = np.einsum("...d,d->...", flap_dir, flow_data.dir) < 0
    down_flap_idxs = np.where(flap_mask)[0][flap_down_mask]

    tris = np.delete(tris, down_flap_idxs, axis=0)
    mesh.tets = mesh.tets[~to_remove_mask]
    end = TriMesh(mesh.nodes, tris)
    edge = end.edge()

    mesh.clean()
    end.clean()
    edge.clean()

    # import pyvista as pv
    # grid = pv.UnstructuredGrid({pv.CellType.TETRA: mesh.tets}, mesh.nodes)
    # bodies = grid.split_bodies()
    # assert(len(bodies) == 1)

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
    # Issue: This can lead to holes in the tri surface. We should detect holes and keep triangles in that case.
    return tris[~surface_mask], normals[~surface_mask]

def remove_duplicate_tris(tris, normals):
    _, inv, cnts = np.unique(tris, axis=0, return_inverse=True, return_counts=True)
    duplicate_mask = cnts[inv] == 2
    return tris[~duplicate_mask], normals[~duplicate_mask]

def calc_normal(tri):
    normal = np.cross(tri[...,1,:]-tri[...,0,:], tri[...,2,:]-tri[...,0,:])
    normal /= np.linalg.norm(normal, axis=-1)[...,None]
    return normal

def calc_overlaps(nodes2d, tris, r):
    tris_flat = nodes2d[tris]
    eps = r/10e6
    overlaps = np.triu(INSCT.tri_intersect2(tris_flat, tris_flat, eps), k=1)
    return np.unique(np.where(overlaps))

def calc_normal_towards(tri, p):
    vec = p-tri[...,0,:]
    normal = np.cross(tri[...,1,:]-tri[...,0,:], tri[...,2,:]-tri[...,0,:])
    normal /= np.linalg.norm(normal, axis=-1)[...,None]
    normal[np.einsum("td,td->t", normal, vec) < 0] *= -1

    return normal

def generate_cluster_mesh(cluster, res, uid):
    centre = np.mean(cluster.V, axis=0)
    sz = 100/np.median(cluster.R)

    V = (cluster.V-centre)*sz
    E = cluster.E
    R = cluster.R*0.65*sz
    radii = ([cluster.in_data.radius] if cluster.in_data is not None else []) + [node.radius for node in cluster.nodes] + [outflow.data.radius for outflow in cluster.outflows]
    dx = ((2*np.min(radii))/res)*sz

    grid = CONV.conv_surf(V, E, R, dx)
    if grid.dim[0]*grid.dim[1]*grid.dim[2] > 250*250*250:
        raise Exception("grid too large")
    verts, tris = contour(grid)

    mesh = run_tetgen(verts, tris, uid, False)

    mesh.nodes = np.float128(mesh.nodes)
    mesh.nodes /= sz
    mesh.nodes += centre
    mesh.nodes = np.float64(mesh.nodes)
    return mesh

def run_tetgen(verts, tris, uid, preserve_surface=True):
    mesh = meshio.Mesh(verts, [("triangle", tris)])
    mesh.write(f"_temp{uid}.mesh")

    options = "-QENFgq1.4/10" + ("Y" if preserve_surface else "")
    _tetgen(options, uid)

    res = meshio.read(f"_temp{uid}.1.mesh")
    if len(res.points) == 0:
        raise Exception("tetgen failed")

    tet = TetMesh(res.points, res.cells_dict["tetra"])
    
    flat_tets = flatness(tet.nodes[tet.tets]) < 2
    flat_tet_tris = np.sort(tet.tets[flat_tets][:,tet_tris], axis=2)
    surf = tet.surface()
    surface_slivers = np.sum(np.any(np.all(surf.tris[:,None, None, :] == flat_tet_tris[None, :, :, :], axis=3),axis=0),axis=1) >= 2
    to_remove = np.arange(len(tet.tets))[flat_tets][surface_slivers]
    tet.tets = np.delete(tet.tets, to_remove, axis=0)

    return tet

def _tetgen(options, uid):
    proc = subprocess.run(["tetgen", options, f"_temp{uid}.mesh"], capture_output=True)
    if (proc.returncode == 0): return

    options = "-QENFgqY"
    proc = subprocess.run(["tetgen", options, f"_temp{uid}.mesh"], capture_output=True)
    if (proc.returncode == 0): return

    options = "-QENFgq"
    proc = subprocess.run(["tetgen", options, f"_temp{uid}.mesh"], capture_output=True)
    if (proc.returncode == 0): return

    raise Exception("tetgen failed")

def compute_cluster_meshes(cluster, res, uid):
    mesh = generate_cluster_mesh(cluster, res, uid)

    out_ends = [None]*len(cluster.outflows)
    for i, outflow in enumerate(cluster.outflows):
        out_ends[i] = process_end(mesh, outflow.data)
    in_end = process_end(mesh, cluster.in_data) if cluster.in_data is not None else None

    return mesh, in_end, out_ends

def end_radius(end):
    _, proj = proj_to_plane(end.edge.nodes, end.flow_data)
    centre = np.mean(proj, axis=0)
    dists = np.linalg.norm(proj-centre, axis=1)
    return np.mean(dists)

def connector_tube(end1, end2, ang_res):
    r1 = end_radius(end1)
    r2 = end_radius(end2)
    l = np.linalg.norm(end1.flow_data.point - end2.flow_data.point)
    frac1 = min(0.1, (0.5*r1)/l)
    # frac1 = min(0.2, (0.5*r1)/l)
    frac2 = 1-min(0.1, (0.5*r2)/l)
    # frac2 = 1-min(0.2, (0.5*r2)/l)
    v_p = lerp(end1.flow_data.point, end2.flow_data.point, frac1)
    v_c = lerp(end1.flow_data.point, end2.flow_data.point, frac2)
    r_p = lerp(r1, r2, frac1)
    r_c = lerp(r1, r2, frac2)
    lin_res = int(np.ceil((np.linalg.norm(v_p-v_c)*ang_res)/((r_p+r_c)*np.pi)/2))
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
    permutation = np.argsort(angle_xy(points))
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

    if (ang_tot > 0):
        permutation = permutation[::-1]
    return points[permutation]

import pyvista as pv
# import triangle as tr
def strip(end, cyl_edge, r):
    Q = QUAT.R_vector_to_vector(end.flow_data.dir, VEC.k())
    def transform(p):     return QUAT.rotate(Q, p-end.flow_data.point)
    def transform_inv(p): return QUAT.rotate(QUAT.conjugate(Q), p)+end.flow_data.point

    cyl_nodes = np.unique(cyl_edge.nodes[cyl_edge.segs].reshape(-1, 3), axis=0)
    end_nodes = end.edge.nodes

    cyl_points = sort_ring_geom(transform(cyl_nodes))
    end_points = sort_ring_morph(transform(end_nodes), end.edge.segs)[::-1, :]

    dists = np.linalg.norm(cyl_points[:,None,:]-end_points[None,:,:],axis=2)
    min_cyl, min_end = np.unravel_index(np.argmin(dists), dists.shape)

    cyl_points = np.roll(cyl_points, -min_cyl, axis=0)
    end_points = np.roll(end_points, -min_end, axis=0)

    cyl_angs = np.zeros(len(cyl_points))
    for i in range(1, len(cyl_points)):
        cyl_angs[i] = cyl_angs[i-1] + angle_in_plane(cyl_points[i-1], cyl_points[i], VEC.k())
    end_angs = np.zeros(len(end_points))
    end_angs[0] = angle_in_plane(cyl_points[0], end_points[0], VEC.k())
    for i in range(1, len(end_points)):
        end_angs[i] = end_angs[i-1] + angle_in_plane(end_points[i-1], end_points[i], VEC.k())
    cyl_heights = -cyl_points[:,2]
    end_heights = -end_points[:,2]
    cyl_projs = np.array([cyl_angs*r, cyl_heights]).T
    end_projs = np.array([end_angs*r, end_heights]).T

    cyl_projs2 = np.vstack((cyl_projs, cyl_projs[0]+np.array([2*np.pi*r,0])))
    end_projs2 = np.vstack((end_projs, end_projs[0]+np.array([2*np.pi*r,0])))

    end_projs_int_idxs = np.arange(len(end_projs2))[1:-1]
    height_order = end_projs_int_idxs[np.argsort(end_projs2[end_projs_int_idxs,1])]

    n = len(cyl_projs2)
    end_to_cyl_map = np.full(len(end_projs2), np.nan)
    def show_map():
        plotter = pv.Plotter()
        points = np.vstack((cyl_projs2, end_projs2))
        points3d = np.hstack((points, np.zeros((len(points),1))))
        plotter.add_point_labels(points3d, [str(i) for i in range(len(points3d))])
        lines = np.array([[points3d[i+n], points3d[int(m)]] for i, m in enumerate(end_to_cyl_map) if np.isfinite(m)])
        plotter.add_lines(lines.reshape(-1,3), color="b")
        plotter.show()
    end_to_cyl_map[0] = 0
    end_to_cyl_map[-1] = len(cyl_projs2)-1
    for i in height_order:
        v1 = end_projs2[i-1]-end_projs2[i]
        v2 = end_projs2[i+1]-end_projs2[i]
        v_cs1 = cyl_projs2 - end_projs2[i-1]
        v_cs2 = cyl_projs2 - end_projs2[i+1]
        left_bound = np.arctan2(np.cross(v_cs1, v1), np.dot(v_cs1, v1)) < -0.4
        right_bound = np.arctan2(np.cross(v_cs2, v2), np.dot(v_cs2, v2)) > 0.4
        valid_idxs = np.where(np.logical_and(np.logical_or(v1[1] > 0, left_bound), np.logical_or(v2[1] > 0, right_bound)))[0]
        left_edges = end_to_cyl_map[:i][::-1]
        left_edge = int(left_edges[np.isfinite(left_edges)][0])
        right_edges = end_to_cyl_map[i+1:]
        right_edge = int(right_edges[np.isfinite(right_edges)][0])
        valid_idxs = valid_idxs[np.logical_and(valid_idxs >= left_edge,valid_idxs <= right_edge)]
        if len(valid_idxs) > 0:
            dists = np.linalg.norm(cyl_projs2 - end_projs2[i], axis=1)
            closest_valid_idx = valid_idxs[np.argmin(dists[valid_idxs])]
            end_to_cyl_map[i] = closest_valid_idx

    unmapped_ranges = []
    cur_range = None
    for i, mapping in enumerate(end_to_cyl_map):
        if np.isfinite(mapping):
            if cur_range is None:
                continue
            else:
                unmapped_ranges.append(cur_range)
                cur_range = None
        else:
            if cur_range is None:
                cur_range = [i]
            else:
                cur_range.append(i)
    # TODO: triangulate unmapped ranges

    extra_tri_meshes = []
    end_nodes = np.vstack((end_points, end_points[0]))
    for unmapped_idxs in unmapped_ranges:
        if len(unmapped_idxs) == 1:
            i = unmapped_idxs[0]
            nodes = end_nodes[[i-1, i, i+1]]
            nodes = transform_inv(nodes)
            tris = np.array([[0,1,2]])
            extra_tri_meshes.append(TriMesh(nodes, tris))
        elif len(unmapped_idxs) == 2:
            idxs = [unmapped_idxs[0]-1] + unmapped_idxs + [unmapped_idxs[len(unmapped_idxs)-1]+1]
            nodes = end_nodes[idxs]
            nodes = transform_inv(nodes)
            tris = np.array([[0,1,2], [2,3,0]])
            extra_tri_meshes.append(TriMesh(nodes, tris))
        else:
            try:
                import triangle as tr
                idxs = [unmapped_idxs[0]-1] + unmapped_idxs + [unmapped_idxs[len(unmapped_idxs)-1]+1]
                t = tr.triangulate({'vertices': end_projs2[idxs]})

                nodes = end_nodes[idxs]
                nodes = transform_inv(nodes)
                tris = t['triangles']

                extra_tri_meshes.append(TriMesh(nodes, tris))
            except:
                raise Exception("too complex unmapped region")

    remaining_end_idxs = np.where(np.isfinite(end_to_cyl_map))[0]
    remaining_end_projs = end_projs2[remaining_end_idxs]
    remaining_end_to_cyl_map = end_to_cyl_map[remaining_end_idxs]

    tris = []
    i = 0
    for i in range(len(remaining_end_projs)-1):
        map1 = int(remaining_end_to_cyl_map[i])
        map2 = int(remaining_end_to_cyl_map[i+1])

        if map1 == map2:
            tris.append([map1, i+n, i+n+1])
        else:
            cyl_between = cyl_projs2[map1:map2+1]
            v1s = remaining_end_projs[i] - cyl_between
            v2s = remaining_end_projs[i+1] - cyl_between
            angs = np.arctan2(np.cross(v2s,v1s), np.einsum("ni,ni->n",v2s,v1s))

            map_mid = np.arange(map1,map2+1)[np.argmax(angs)]

            for mapi in range(map1,map_mid):
                tris.append([mapi, mapi+1, i+n])
            tris.append([map_mid, i+n, i+n+1])
            for mapi in range(map_mid,map2):
                tris.append([mapi, mapi+1, i+n+1])
    tris = np.array(tris)
    # all_p = np.vstack((cyl_projs2, end_projs2[::-1,:]))
    # lines = np.vstack((np.arange(all_p.shape[0]), np.roll(np.arange(all_p.shape[0]),-1))).T
    # lines = np.concatenate(([all_p.shape[0]+1], np.arange(all_p.shape[0]), [0]))
    # contour = pv.PolyData(all_p, faces=lines)
    # flat_tris = contour.delaunay_2d(edge_source=contour)
    # tris = tr.triangulate(dict(vertices=all_p, segments=lines), opts="p")['triangles']
    # tris = flat_tris.faces.reshape(-1, 4)[:,1:]
    nodes = np.vstack((cyl_points, cyl_points[0], end_nodes[remaining_end_idxs]))
    nodes = transform_inv(nodes)
    strip_mesh = TriMesh(nodes, tris)

    def show_tris():
        plotter = pv.Plotter()
        points = np.vstack((cyl_projs2, end_projs2))
        points3d = np.hstack((points, np.zeros((len(points),1))))
        plotter.add_point_labels(points3d, [str(i) for i in range(len(points3d))])
        plotter.add_mesh(pv.PolyData(points3d, np.concatenate((np.full((tris.shape[0], 1), 3), tris), axis=1)), show_edges=True)
        plotter.show()

    if len(extra_tri_meshes) > 0:
        strip_mesh = merge_tri_meshes([strip_mesh] + extra_tri_meshes)
    return strip_mesh
import pymeshfix
def make_connector(end1: EndSlice, end2: EndSlice, uid) -> TetMesh:
    edge_cnt = int(np.ceil((end1.edge.segs.shape[0]+end2.edge.segs.shape[0])/2))

    conn_mesh, conn_edge1, conn_edge2 = connector_tube(end1, end2, edge_cnt)

    strip1 = strip(end1, conn_edge1, 0.95*end1.flow_data.radius)
    strip2 = strip(end2, conn_edge2, 0.95*end2.flow_data.radius)

    connector_tri = merge_tri_meshes([end1.end, strip1, conn_mesh, strip2, end2.end])

    centre = np.mean(connector_tri.nodes, axis=0)
    sz = 100/np.median([end1.flow_data.radius, end2.flow_data.radius])

    verts = (connector_tri.nodes-centre)*sz
    tris = connector_tri.tris
    verts, tris = pymeshfix.clean_from_arrays(verts, tris)

    tet = run_tetgen(verts, tris, uid)

    tet.nodes = np.float128(tet.nodes)
    tet.nodes /= sz
    tet.nodes += centre
    tet.nodes = np.float64(tet.nodes)

    return tet