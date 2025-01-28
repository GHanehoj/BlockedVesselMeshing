import data as DATA
import tree as TREE
import clusters as CLUSTERS
import full_generation as GEN
# from tools.pyvista_plotting import show_tet_mesh
from tools.mesh_util import TetMesh
from tqdm import tqdm
_MAX_DEPTH = 1000

tree_folder = f"../data/trees/__reg200"
V, E, R = DATA.load_skeleton_data(tree_folder)
root, _ = TREE.make_tree(V, E, R)
root.position += 2*(root.position - root.children[0].position)

root_cluster = CLUSTERS.make_cluster(root.children[0])

cluster = root_cluster

total_cnt = CLUSTERS._cnt(cluster, 0, _MAX_DEPTH)

pbar = tqdm(total=total_cnt)
def done_f():
    pbar.update(1)

res = GEN.gen_tree_clustered(cluster, done_f, _MAX_DEPTH)

tet = TetMesh(res.nodes, res.tets)


print(tet.size(), "MB")
tet.save("full_3_2.mesh")
# show_tet_mesh(tet)
