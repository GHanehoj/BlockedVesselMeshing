import numpy as np
import tetgen
import rainbow.math.quaternion as QUAT
import rainbow.math.vector3 as VEC
from tools.mesh_util import SegMesh, TriMesh, TetMesh, surface_edge
from tools.contouring import branch_contour
from tools.numpy_util import ang_mod, angle_in_plane, angle_xy, lerp
from collections import namedtuple
EndSlice = namedtuple("EndSlice", ["end", "edge", "point", "dir", "radius"])

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
    mask = np.dot(nodes-point, dir) > 0.0001
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
        # def show():
        #     import pyvista as pv
        #     from tools.pyvista_plotting import add_tri_mesh, add_tet_mesh
        #     plt = pv.Plotter()
        #     add_tet_mesh(plt, TetMesh(nodes, elems))
        #     add_tri_mesh(plt, end)
        #     plt.show()
        Q = QUAT.R_vector_to_vector(dir, VEC.k())
        def transform(p):     return QUAT.rotate(Q, p-point)
        def transform_inv(p): return QUAT.rotate(QUAT.conjugate(Q), p)+point
        
        _, sorted_idxs = sort_ring_morph(transform(nodes), edge.segs)

        n = len(sorted_idxs)
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        theta0 = angle_xy(transform(nodes[sorted_idxs[0]]))
        x = radius * np.cos(theta-theta0)
        y = radius * np.sin(theta-theta0)
        z = np.zeros(n)

        circle_nodes = np.dstack((x,y,z))
        circle_nodes = circle_nodes.reshape(-1, 3)
        circle_nodes = transform_inv(circle_nodes)

        nodes[sorted_idxs] = circle_nodes

        # neighbours = mk_neigbours_tri(nodes, end.tris)
        # laplacian_smoothing(nodes, neighbours, np.logical_or(~mk_mask(np.unique(end.tris), len(nodes)), mk_mask(b, len(nodes))), lr=0.1, iter=50)

        ends[i] = EndSlice(end, edge, point, dir, radius)
        elems = remove_tip(nodes, elems, point, dir)

    return TetMesh(nodes, elems), ends

def connector_tube(end1, end2, ang_res):
    v_p = lerp(end1.point, end2.point, 0.1)
    v_c = lerp(end1.point, end2.point, 0.9)
    r_p = lerp(end1.radius, end2.radius, 0.1)
    r_c = lerp(end1.radius, end2.radius, 0.9)
    lin_res = int(np.ceil((np.linalg.norm(v_p-v_c)*ang_res)/((r_p+r_c)*np.pi)/4))
    return tube(v_p, v_c, r_p*0.9, r_c*0.9, ang_res, lin_res)

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

    #     plotter.show()
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
