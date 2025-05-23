from blocked import compute_cluster_meshes, make_connector
from tools.mesh_util import MultiTetMesh

def gen_tree_clustered(root_cluster, res, done_f, max_depth, uid):
    multi_tet, _, fail_cnt = gen_node_clustered_rec(root_cluster, res, done_f, 0, max_depth, uid)
    return multi_tet, fail_cnt

def gen_node_clustered_rec(cluster, res, done_f, depth, max_depth, uid):
    cluster_tet, cluster_in_end, cluster_out_ends = compute_cluster_meshes(cluster, res, uid)
    done_f()

    if depth > max_depth: return MultiTetMesh(cluster_tet, [], [], [], []), cluster_in_end, 0

    child_tets       = []
    connector_tets   = []
    child_in_ends    = []
    used_out_ends    = []

    fail_cnt = 0

    for i in range(len(cluster.outflows)):
        try:
            child_tet, child_in_end, child_fails = gen_node_clustered_rec(cluster.outflows[i].cluster, res, done_f, depth+1, max_depth, uid)
            connector_tet = make_connector(cluster_out_ends[i], child_in_end, uid)

            child_tets.append(child_tet)
            connector_tets.append(connector_tet)
            child_in_ends.append(child_in_end)
            used_out_ends.append(cluster_out_ends[i])

            fail_cnt += child_fails
        except Exception as e:
            print(f"cluster with root {cluster.nodes[0].index} failed: {e}")
            fail_cnt += 1
    return MultiTetMesh(cluster_tet, child_tets, connector_tets, used_out_ends, child_in_ends), cluster_in_end, fail_cnt
