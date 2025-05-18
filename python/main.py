import sys
sys.setrecursionlimit(10**6)
import load as LOAD
import tree as TREE
import clusters as CLUSTERS
import full_generation as GEN
from tools.pyvista_plotting import show_tet_mesh
from tools.mesh_util import TetMesh
from tools.numpy_util import mk_mask
from tqdm import tqdm
import numpy as np
np.seterr(divide="raise", invalid="raise")
_MAX_DEPTH = 1000

# tree_folder = f"../data/trees/__reg200"
# V, E, R = LOAD.load_skeleton_data(tree_folder)
# root = LOAD.load_vesselgraph_data("../data/input/brain/VesselGraph/synthetic_graph_1/1_b_3_0/1_b_3_0_nodes_processed.csv",
#                                   "../data/input/brain/VesselGraph/synthetic_graph_1/1_b_3_0/1_b_3_0_edges_processed.csv")
# root = LOAD.load_hepatic_vtk("../data/input/liver/")
# root = LOAD.load_adtree("../data/input/tree/Lille_11.ply", 0.08)
root = LOAD.load_adtree("../data/input/tree/tree33.ply", 0.08)
# root = LOAD.load_lsystem("../data/input/tree/sympodial/")
# root = LOAD.load_segmented_curves("../data/input/lung/lung_curves_good/", 0.08)

pbar = tqdm(total=TREE.size(root))
def done_f(): pbar.update(1)
root_cluster = CLUSTERS.make_cluster(root.children[0], done_f)

stats = CLUSTERS.cluster_stats(root_cluster, 4)
huge = np.where(stats > 1000000)[0]
if len(huge) > 0:
    print("Huge clusters found:", huge)

pbar = tqdm(total=CLUSTERS._cnt(root_cluster, 0, _MAX_DEPTH))
def done_f(): pbar.update(1)

res = GEN.gen_tree_clustered(root_cluster, done_f, _MAX_DEPTH)

tet = TetMesh(res.nodes, res.tets)


print(tet.size(), "MB")
show_tet_mesh(tet)
# tet.save("/media/data/data/meshes/brain/v2.mesh")
# tet.save("../data/meshes/liver_v3.mesh")
