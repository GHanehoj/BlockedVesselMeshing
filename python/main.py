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
import time
import matplotlib.pyplot as plt
from tools.pyvista_plotting import *
np.seterr(divide="raise", invalid="raise")
_MAX_DEPTH = 100000
res = 4

experiment = int(sys.argv[1])
if experiment == 0:
    example = "kidney"
    sub_example = ""
if experiment == 1:
    example = "brain"
    sub_example = "_vg"
if experiment == 2:
    example = "liver"
    sub_example = "_pp"
if experiment == 3:
    example = "lung"
    sub_example = ""
if experiment == 4:
    example = "tree"
    sub_example = "_ahn"

## Example 1: Kidney
if example == "kidney":
    root = LOAD.load_skeleton(f"../data/input/kidney/__reg200")

## Example 2: Synthetic brain (DeepVesselNet)
if example == "brain":
    if sub_example == "_dpn":
      root = LOAD.load_vesselgraph_data("../data/input/brain/DeepVesselNet/nodes.csv",
                                      "../data/input/brain/DeepVesselNet/edges.csv")
    if sub_example == "_vg":
    #   root = LOAD.load_vesselgen_data("../data/input/brain/VesselGen/result_3/test_1_result_12", 0.08)
        root = LOAD.load_skeleton(f"../data/input/brain/skeleton")

## Example 3: Synthetic Liver
if example == "liver":
    # root = LOAD.load_hepatic_vtk("../data/input/liver/")
    if sub_example == "":
        root = LOAD.load_skeleton(f"../data/input/liver/skeleton")
    if sub_example == "_pp":
        root = LOAD.load_skeleton(f"../data/input/liver/preprocessed/reg25")

## Example 4: Segmented Lung
if example == "lung":
    # root = LOAD.load_segmented_curves("../data/input/lung/lung_curves_good/", 0.08)
    root = LOAD.load_skeleton(f"../data/input/lung/skeleton")

## Examples 5: L-system Trees
if example == "tree":
    if sub_example == "_ahn":
        # root = LOAD.load_adtree("../data/input/tree/ahn/ahn3_delft.ply", 0.08)
        root = LOAD.load_skeleton(f"../data/input/tree/skeleton")
    if sub_example == "_Lille":
        root = LOAD.load_adtree("../data/input/tree/lille/Lille_11.ply", 0.08)

pbar = tqdm(total=TREE.size(root))
def done_f(): pbar.update(1)
t0 = time.time()
root_cluster = CLUSTERS.make_cluster(root, done_f)

stats = CLUSTERS.cluster_stats(root_cluster, res)
huge = np.where(stats[:,0] > 100000)[0]
if len(huge) > 0:
    print("Huge clusters found:", huge)

pbar = tqdm(total=CLUSTERS._cnt(root_cluster, 0, _MAX_DEPTH))
def done_f(): pbar.update(1)

multi_tet, fail_cnt = GEN.gen_tree_clustered(root_cluster, res, done_f, _MAX_DEPTH, experiment)
t1 = time.time()
tet = TetMesh(multi_tet.nodes, multi_tet.tets)


print(tet.size(), "MB")
print(t1-t0)


# file = f"/media/data/data/meshes/{example}/{example}{sub_example}_{res}"
file = f"../data/meshes/{example}{sub_example}_{res}"
with open(file+".txt", "w") as f:
  f.write(f"example: {example}{sub_example}, res: {res}, size: {tet.size()} MB, time: {t1-t0} ms, err: {fail_cnt}")
tet.save(file+".mesh")
# show_tet_mesh(tet)
# tet.save("../data/meshes/liver_v3.mesh")