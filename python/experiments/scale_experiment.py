import sys
import os
sys.path.append(os.path.abspath('../'))
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
np.seterr(divide="raise", invalid="raise")
_MAX_DEPTH = 100000

def gen_skeleton2(n_segs):
    r0 = 0.2
    frac = 0.85
    V = [[0,0], [1,0]]
    E = [[0,1]]
    R = [r0, r0]

    def rec(l, r, p_idx, cnt):
        if cnt >= n_segs: return
        l = frac*l
        r = frac*r

        p_v = V[p_idx]

        c2_v = p_v + np.array([1,0])*l
        c2_idx = len(V)
        V.append(c2_v)
        E.append([p_idx, c2_idx])
        R.append(r)
        rec(l, r, c2_idx, cnt+1)

    rn = r0 * frac**n_segs
    rec(1, r0, 1, 1)
    dx = ((2*rn)/10)
    return np.hstack((np.array(V), np.zeros((len(V),1)))), np.array(E), np.array(R), dx

def run(V, E, R):
    root, _ = TREE.make_tree_unordered2(V, E, R)

    pbar = tqdm(total=TREE.size(root))
    def done_f(): pbar.update(1)
    t0 = time.time()
    root_cluster = CLUSTERS.make_cluster(root.children[0], done_f)

    pbar = tqdm(total=CLUSTERS._cnt(root_cluster, 0, _MAX_DEPTH))
    def done_f(): pbar.update(1)

    multi_tet, fail_cnt = GEN.gen_tree_clustered(root_cluster, 10, done_f, _MAX_DEPTH)
    t1 = time.time()
    return t1-t0


# ts = []
# ns = []

# for n in range(5,18):
#     V, E, R, _ = gen_skeleton2(n)
    
#     t = run(V, E, R)
#     ts.append(t)
#     ns.append(n)

# ts = np.array(ts)
# ns = np.array(ns)


ns = np.load("ns.npy")
ts = np.load("ts.npy")
ns_global = np.load("ns_global.npy")
ts_global = np.load("ts_global.npy")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.set_context("talk")
plt.plot(ns, ts/ns,marker="o")
plt.plot(ns_global, ts_global/ns_global,marker="o")
plt.xlabel("# of segments")
plt.ylabel("time per segment (s)")
plt.legend(['Blocked Convolution', 'Global Convolution'])
plt.show()
# np.save("ts.npy", ts)
# np.save("ns.npy", ns)
