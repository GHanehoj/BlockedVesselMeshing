import sys
import os
sys.path.append(os.path.abspath('../'))
sys.setrecursionlimit(10**6)
import load as LOAD
import tree as TREE
import clusters as CLUSTERS
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.pyvista_plotting import *
np.seterr(divide="raise", invalid="raise")
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context("talk")

examples = ["brain/VesselGen/result_3"]
ns = [n for n in range(96) if (n < 5 or (n < 100 and n % 5 == 0) or n % 50 == 0)]

# brain = np.empty((len(examples), len(ns), 2))

# for i, example in enumerate(examples):
#     for j, n in enumerate(tqdm(ns)):
#         tree_folder = f"../../data/input/{example}/preprocessed/reg{n}"
#         V, E, R = LOAD.load_skeleton_data(tree_folder)
#         root, _ = TREE.make_tree(V, E, R)

#         root_cluster = CLUSTERS.make_cluster(root.children[0])

#         stats = CLUSTERS.cluster_stats(root_cluster, 3)

#         brain[i,j,0] = np.mean(stats[:,0])
#         brain[i,j,1] = np.mean(stats[:,1])


# np.save("all_outcomes.npy", all)
all = np.load("all_outcomes.npy")
all_normalized_mesh_sizes = all[:,:,0] / all[:,0,0][:,None]*100
names = ["Brain", "Lung", "Tree", "Kidney", "Liver"]
for i, name in enumerate(names):
    plt.plot(ns, all_normalized_mesh_sizes[i, :len(ns)], label=name)
plt.xlabel("# of iterations")
plt.ylabel("Mean cluster mesh size (normalized)")
plt.ylim(0,120)
plt.xlim(0,47)
plt.legend()
plt.show()

# all_normalized_cluster_sizes = (all[:,:,1]-1) / (all[:,0,1][:,None]-1)*100
for i, name in enumerate(names):
    plt.plot(ns, all[i, :len(ns), 1], label=name)

plt.xlabel("# of iterations")
plt.ylabel("Mean cluster node count (normalized)")
# plt.ylim(0,110)
# plt.xlim(0,47)
plt.legend()
plt.show()
