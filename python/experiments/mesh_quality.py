import sys
import os
sys.path.append(os.path.abspath('../'))
from tools.mesh_util import load_tet
import numpy as np
from rainbow.math.tetrahedron import *
from tqdm import tqdm
edge_idxs = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])
tet = load_tet("/media/data/data/meshes/meshes/lung_3.mesh")

# tet_sample = np.random.choice(np.arange(len(tet.tets)), 1000000)
r_outs = np.array([compute_circumscribed_sphere(*tet)[1] for tet in tqdm(tet.nodes[tet.tets])])
r_ins = np.array([compute_inscribed_sphere(*tet)[1] for tet in tqdm(tet.nodes[tet.tets])])
# vols = np.array([abs(compute_signed_volume(*tet)) for tet in tqdm(tet.nodes[tet.tets[tet_sample]])])
# l2s = np.array([np.sum(np.square(np.linalg.norm(tet[edge_idxs][:,1]-tet[edge_idxs][:,0], axis=1))) for tet in tqdm(tet.nodes[tet.tets[tet_sample]])])

rad_ratio = 3*r_ins/r_outs
np.save("rad_ratio.npy", rad_ratio)

# mean_ratio = 12*np.power(3*vols, 2/3)/l2s
# np.save("mean_ratio.npy", mean_ratio)

# rad_ratio = np.load("../../data/tmp/rad_ratio.npy")
# mean_ratio = np.load("../../data/tmp/mean_ratio.npy")


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid", font_scale = 2)
def plot(data, x_label):
    fig,ax = plt.subplots(figsize=(7,4))
    sns.histplot(data, ax=ax, stat="density", bins=50)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    plt.show()

plot(rad_ratio, "Radius Ratio ($3\\frac{R_{in}}{R_{out}}$)")
