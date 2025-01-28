from blocked import compute_cluster_meshes, make_connector
from tools.mesh_util import MultiTetMesh

def gen_tree_clustered(root_cluster, done_f, max_depth):
    return gen_node_clustered_rec(root_cluster, done_f, 0, max_depth)[0]

def gen_node_clustered_rec(cluster, done_f, depth, max_depth):
    cluster_tet, cluster_in_end, cluster_out_ends = compute_cluster_meshes(cluster)

    if depth > max_depth: return MultiTetMesh(cluster_tet, [], [], [], []), cluster_in_end

    child_tets       = [None]*len(cluster.outflows)  #: List['MultiTetMesh'],
    connector_tets   = [None]*len(cluster.outflows)  #: List[TetMesh],
    child_in_ends    = [None]*len(cluster.outflows)  #: List[EndSlice]):

    for i in range(len(cluster.outflows)):
        try:
            child_tets[i], child_in_ends[i] = gen_node_clustered_rec(cluster.outflows[i].cluster, done_f, depth+1, max_depth)
            connector_tets[i] = make_connector(cluster_out_ends[i], child_in_ends[i])
        except Exception as e:
            print(f"cluster with root {cluster.nodes[0].index} failed")
            return MultiTetMesh(cluster_tet, [], [], [], []), cluster_in_end
    done_f()
    return MultiTetMesh(cluster_tet, child_tets, connector_tets, cluster_out_ends, child_in_ends), cluster_in_end
