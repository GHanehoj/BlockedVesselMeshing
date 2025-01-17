import numpy as np
import tetgen
from blocked import compute_cluster_meshes, connector_tube, strip
from tools.numpy_util import mk_mask
from tools.mesh_util import MultiTetMesh, TetMesh, clean_tet_mesh, merge_tri_meshes
from tools.smoothing import laplacian_smoothing, taubin_smoothing
_meshfails = 0
_connfails = 0

def smooth_transition(tet_mesh, parent_end):
    p0 = parent_end.point-parent_end.dir*parent_end.radius
    p1 = parent_end.point+parent_end.dir*parent_end.radius

    parent_mask = np.dot(tet_mesh.nodes-p0, parent_end.dir) < 0
    child_mask = np.dot(tet_mesh.nodes-p1, -parent_end.dir) < 0
    geom_mask = np.logical_or(parent_mask, child_mask)

    ## First, we smooth the surface.
    surface = tet_mesh.surface()
    surface_neigh = surface.calc_neighbours()
    taubin_smoothing(tet_mesh.nodes, surface_neigh, geom_mask, iter=10)

    ## Then the volume
    fixed_mask = np.logical_or(geom_mask, mk_mask(np.unique(surface.tris), len(tet_mesh.nodes)))
    volume_neigh = tet_mesh.calc_neighbours()
    # volume_neigh = tet_mesh.neigh_raw[~fixed_mask]/np.maximum(1, tet_mesh.neigh_tots[~fixed_mask,None])
    laplacian_smoothing(tet_mesh.nodes, volume_neigh, fixed_mask)

def gen_tree_clustered(root_cluster, done_f, max_depth):
    global _meshfails
    global _connfails
    _meshfails = 0
    _connfails = 0

    res = MultiTetMesh()

    gen_node_clustered_rec(res, root_cluster, done_f, None, None, 0, max_depth)

    return res

def gen_node_clustered_rec(res, cluster, done_f, parent_end, parent_id, depth, max_depth):
    if depth > max_depth: return

    try:
        cluster_tet, in_end, out_ends = compute_cluster_meshes(cluster)
        cluster_id = res.append_mesh(cluster_tet)
    except:
        done_f()
        print(f"cluster with root {cluster.nodes[0].index} failed mesh generation")
        global _meshfails
        _meshfails += 1
        for i in range(len(cluster.outflows)):
            gen_node_clustered_rec(res, cluster.outflows[i].cluster, done_f, None, None, depth+1)
        return

    if parent_end is not None:
        edge_cnt = int(np.ceil((in_end.edge.segs.shape[0]+parent_end.edge.segs.shape[0])/2))

        conn_mesh, conn_edge1, conn_edge2 = connector_tube(parent_end, in_end, edge_cnt)

        strip1 = strip(parent_end, conn_edge1)
        strip2 = strip(in_end, conn_edge2)

        connector_tri = merge_tri_meshes([in_end.end, strip1, conn_mesh, strip2, parent_end.end])

        try:
            tgen = tetgen.TetGen(connector_tri.nodes, connector_tri.tris)
            nodes, elems = tgen.tetrahedralize(order=1, mindihedral=20, minratio=1.5, nobisect=True)
            connector_tet = TetMesh(nodes, elems)
        except:
            done_f()
            print(f"cluster with root {cluster.nodes[0].index} failed conector generation")
            global _connfails
            _connfails += 1
            for i in range(len(cluster.outflows)):
                gen_node_clustered_rec(res, cluster.outflows[i].cluster, done_f, out_ends[i], cluster_id, depth+1)
            return

        conn_id = res.append_mesh(connector_tet)

        sub_mesh = res.get_sub_mesh([parent_id, conn_id, cluster_id])

        smooth_transition(sub_mesh, parent_end)
        smooth_transition(sub_mesh, in_end)

        res.write_back(sub_mesh, [parent_id, conn_id, cluster_id])

    done_f()
    for i in range(len(cluster.outflows)):
        gen_node_clustered_rec(res, cluster.outflows[i].cluster, done_f, out_ends[i], cluster_id, depth+1, max_depth)
