import sys
sys.setrecursionlimit(10**6)
import preprocessing.formats as FORMATS
import tree as TREE
import clusters as CLUSTERS
import full_generation as GEN
from tools.mesh_util import TetMesh
from tqdm import tqdm
import numpy as np
import time
from tools.pyvista_plotting import *
np.seterr(divide="raise", invalid="raise")
_MAX_DEPTH = 100000
res = 4

experiment = int(sys.argv[1])
if experiment == 0:
    example = "kidney"
if experiment == 1:
    example = "brain"
if experiment == 2:
    example = "liver"
if experiment == 3:
    example = "lung"
if experiment == 4:
    example = "tree"

## Example 1: Kidney
if example == "kidney":
    root = FORMATS.load_skeleton_data("../data/input/kidney/skeleton")

## Example 2: Synthetic brain (DeepVesselNet)
if example == "brain":
    root = FORMATS.load_skeleton_data(f"../data/input/brain/skeleton")

## Example 3: Synthetic Liver
if example == "liver":
    root = FORMATS.load_skeleton_data(f"../data/input/liver/skeleton")

## Example 4: Segmented Lung
if example == "lung":
    root = FORMATS.load_skeleton_data(f"../data/input/lung/skeleton")

## Examples 5: L-system Trees
if example == "tree":
    root = FORMATS.load_skeleton_data(f"../data/input/tree/skeleton")

pbar = tqdm(total=TREE.size(root))
def done_f(): pbar.update(1)
t0 = time.time()
root_cluster = CLUSTERS.make_cluster(root, done_f)

stats = CLUSTERS.cluster_stats(root_cluster, res)
huge = np.where(stats[:,0] > 100000)[0]
if len(huge) > 0:
    print("Huge clusters found:", huge)

pbar = tqdm(total=CLUSTERS.count_nodes(root_cluster, 0, _MAX_DEPTH))
def done_f(): pbar.update(1)

multi_tet, fail_cnt = GEN.gen_tree_clustered(root_cluster, res, done_f, _MAX_DEPTH, experiment)
t1 = time.time()
tet = TetMesh(multi_tet.nodes, multi_tet.tets)

print(tet.size(), "MB")
print(t1-t0)

file = f"../data/meshes/{example}_{res}"
with open(file+".txt", "w") as f:
  f.write(f"example: {example}, res: {res}, size: {tet.size()} MB, time: {t1-t0} ms, err: {fail_cnt}")
tet.save(file+".mesh")

show_tet_mesh(tet)

debug=1