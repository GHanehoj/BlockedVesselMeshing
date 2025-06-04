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

root = LOAD.load_skeleton(f"../data/input/brain/VesselGen/result_3/skeleton")
root_cluster = CLUSTERS.make_cluster(root, lambda:{})
stats = CLUSTERS.cluster_stats(root_cluster, res)

root = LOAD.load_skeleton(f"../data/input/liver/skeleton")
root_cluster = CLUSTERS.make_cluster(root, lambda:{})
stats = CLUSTERS.cluster_stats(root_cluster, res)

root = LOAD.load_skeleton(f"../data/input/lung/skeleton")
root_cluster = CLUSTERS.make_cluster(root, lambda:{})
stats = CLUSTERS.cluster_stats(root_cluster, res)

root = LOAD.load_skeleton(f"../data/input/tree/ahn/skeleton")
root_cluster = CLUSTERS.make_cluster(root, lambda:{})
stats = CLUSTERS.cluster_stats(root_cluster, res)
a=2