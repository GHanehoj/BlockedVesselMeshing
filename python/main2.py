import sys
sys.setrecursionlimit(10**6)
import tree as TREE
import clusters as CLUSTERS
import full_generation as GEN
from tools.pyvista_plotting import *
from tools.mesh_util import TetMesh
from tools.numpy_util import mk_mask
import tools.file as FILE
from tqdm import tqdm
import numpy as np
import time
import pyvista as pv
from tools.pyvista_plotting import *
np.seterr(divide="raise", invalid="raise")
_MAX_DEPTH = 10000

res = 3
example = "lung"
sub_example = ""

## Example 1: Kidney
if example == "kidney":
    tree_folder = f"../data/input/kidney/preprocessed/old/__reg200"
    V, E, R = FILE.load_skeleton_data(tree_folder)
    root, _ = TREE.make_tree(V, E, R)

## Example 2: Synthetic brain (DeepVesselNet)
# if example == "brain":
#     root = LOAD.load_vesselgraph_data("../data/input/brain/DeepVesselNet/nodes.csv",
#                                       "../data/input/brain/DeepVesselNet/edges.csv")

# ## Example 3: Synthetic Liver
# if example == "liver":
#     root = LOAD.load_hepatic_vtk("../data/input/liver/")

## Example 4: Segmented Lung
if example == "lung":
    V, E, R = FILE.load_skeleton_data("../data/input/lung/skeleton")
    root, _ = TREE.make_tree(V, E, R)

## Examples 5: L-system Trees
# if example == "tree":
#     if sub_example == "_ahn":
#         root = LOAD.load_adtree("../data/input/tree/ahn/ahn3_delft.ply", 0.08)
#     if sub_example == "_Lille":
#         root = LOAD.load_adtree("../data/input/tree/lille/Lille_11.ply", 0.08)


pbar = tqdm(total=TREE.size(root))
def done_f(): pbar.update(1)
t0 = time.time()
root_cluster = CLUSTERS.make_cluster(root, done_f)

stats = CLUSTERS.cluster_stats(root_cluster, res)
huge = np.where(stats[:,0] > 100000)[0]
if len(huge) > 0:
    print("Huge clusters found:", huge)
print(CLUSTERS.count_nodes(root_cluster, 0, _MAX_DEPTH))

# show_clusters(CLUSTERS.cluster_list(root_cluster))

pbar = tqdm(total=CLUSTERS.count_nodes(root_cluster, 0, _MAX_DEPTH))
def done_f(): pbar.update(1)


multi_tet, fail_cnt = GEN.gen_tree_clustered(root_cluster, res, done_f, _MAX_DEPTH, "main2")
t1 = time.time()
tet = TetMesh(multi_tet.nodes, multi_tet.tets)


print(tet.size(), "MB")
print(t1-t0)


# file = f"/media/data/data/meshes/{example}/{example}{sub_example}_{res}"
file = f"../data/meshes/main2_1"
with open(file+".txt", "w") as f:
  f.write(f"example: {example}{sub_example}, res: {res}, size: {tet.size()} MB, time: {t1-t0} ms, err: {fail_cnt}")
tet.save(file+".mesh")
pl = pv.Plotter()
add_tet_mesh(pl, tet)
add_graph(pl, V, E, R)
for i, cluster in enumerate(CLUSTERS.cluster_list(root_cluster)):
    pl.add_point_labels(cluster.nodes[0].position, [str(i)])
pl.show()
# tet.save("../data/meshes/liver_v3.mesh")

a=2