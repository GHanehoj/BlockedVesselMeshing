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

res = 3
example = "liver"
sub_example = ""

## Example 1: Kidney
if example == "kidney":
    tree_folder = f"../data/input/kidney/preprocessed/old/__reg200"
    V, E, R = LOAD.load_skeleton_data(tree_folder)
    root, _ = TREE.make_tree(V, E, R)

## Example 2: Synthetic brain (DeepVesselNet)
if example == "brain":
    root = LOAD.load_vesselgraph_data("../data/input/brain/DeepVesselNet/nodes.csv",
                                      "../data/input/brain/DeepVesselNet/edges.csv")

## Example 3: Synthetic Liver
if example == "liver":
    root = LOAD.load_hepatic_vtk("../data/input/liver/")

## Example 4: Segmented Lung
if example == "lung":
    root = LOAD.load_segmented_curves("../data/input/lung/lung_curves_good/", 0.08)

## Examples 5: L-system Trees
if example == "tree":
    if sub_example == "_ahn":
        root = LOAD.load_adtree("../data/input/tree/ahn/ahn3_delft.ply", 0.08)
    if sub_example == "_Lille":
        root = LOAD.load_adtree("../data/input/tree/lille/Lille_11.ply", 0.08)
