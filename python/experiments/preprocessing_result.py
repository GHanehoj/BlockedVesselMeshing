import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import data as DATA
import tree as TREE
from scipy.stats import gaussian_kde
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
sns.set_style('whitegrid')

def calc_arm_lens(nodes):
    return np.array([[np.linalg.norm(node.children[i].position-node.position) for i in [0,1]] for node in nodes])

def angle_between(v1, v2):
    v1_u = v1/np.linalg.norm(v1)
    v2_u = v2/np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def calc_arm_colinearity(nodes):
    angles = np.empty((len(nodes), 2))
    for i, node in enumerate(nodes):
        a_in = node.parent.position - node.position
        a_out1 = node.children[0].position - node.position
        a_out2 = node.children[1].position - node.position
        angles[i, 0] = angle_between(a_in, a_out1)
        angles[i, 1] = angle_between(a_in, a_out2)
    return angles

def plot_smallest_arm_lengths(branches):
    arm_lens = calc_arm_lens(branches)
    child_lens = arm_lens[:, 1:].flatten()
    fig, ax = plt.subplots(figsize=(7, 5))

    kde = gaussian_kde(child_lens)
    x = np.linspace(0, 3, 100)
    y = kde(x)
    area = simpson(y, x, dx=0.001) * 100

    sns.kdeplot(data=child_lens, ax=ax, clip=(0, 50))
    ax.fill_between(x=x, y1=y, color='r', alpha=.5)

    ax.annotate(f'{area:.2f}%', xy=(1.5, 0.003), xytext=(5, 0.005), arrowprops=dict(arrowstyle="->", color='r', alpha=.5))
    ax.set_xlabel("Normalized arm length")
    ax.set_xticks([0, 3, 10, 20, 30, 40, 50])
    ax.set_xticklabels(["0", "3", "10", "20", "30", "40", "50"])
    ax.set_title(f"Distribution of arm lengths (max={np.max(child_lens):.1f})")
    return fig

def plot_outflow_angles(branches):
    angles = calc_arm_colinearity(branches)
    child_angles = angles[:, [0,2]].flatten()
    fig, ax = plt.subplots(figsize=(7, 5))

    kde = gaussian_kde(child_angles)
    x = np.linspace(0, np.pi/10, 100)
    y = kde(x)
    area = simpson(y, x, dx=0.001) * 100

    ax.fill_between(x=x, y1=y, color='r', alpha=.5)
    ax.annotate(f'{area:.2f}%', xy=(0.1, 0.05), xytext=(0.2, 0.3), arrowprops=dict(arrowstyle="->", color='r', alpha=.5))
    sns.kdeplot(data=angles[:, 0], label="Large arm", ax=ax, clip=(0.0, np.pi))
    sns.kdeplot(data=angles[:, 2], label="Small arm", ax=ax, clip=(0.0, np.pi))
    ax.set_xlabel("Angle to inflow (radians)")
    ax.set_title("Distribution of outflow angles")
    ax.legend()
    return fig

# branches = BRANCH.get_all_branches(TreeConf("orig", True))
# fig = plot_outflow_angles(branches)
# # fig.savefig("outflow_angles.png")
# plt.show()

# fig = plot_smallest_arm_lengths(branches)
# # fig.savefig("outflow_angles.png")
# plt.show()


def calc_diagnositcs(xs):
    tree_folder = f"../../data/trees/split"
    V, E, R = DATA.load_skeleton_data(tree_folder)
    _, nodes = TREE.make_tree(V, E, R)
    nodes = [node for node in nodes if len(node.children) == 2 and node.parent is not None]
    V0 = V
    dists = np.empty((len(xs), len(V0)))
    arm_lens = np.empty((len(xs), len(nodes), 2))
    angles = np.empty((len(xs), len(nodes), 2))
    for i, x in enumerate(tqdm(xs)):
        tree_folder = f"../../data/trees/__reg{x}"
        V, E, R = DATA.load_skeleton_data(tree_folder)
        _, nodes = TREE.make_tree(V, E, R)
        nodes = [node for node in nodes if len(node.children) == 2 and node.parent is not None]

        dists[i] = np.linalg.norm(V-V0, axis=1)

        arm_lens[i] = calc_arm_lens(nodes)
        angles[i] = calc_arm_colinearity(nodes)

    return dists, arm_lens, angles
    


barrier_vals = np.loadtxt("../../data/tmp/barrier_vals.npy")
xs = np.array([x for x in range(501) if x < 5 or (x < 100 and x % 5 == 0) or x % 50 == 0])
# dists, arm_lens, angles = calc_diagnositcs(xs)

# np.save("../../data/tmp/dists.npy", dists)
# np.save("../../data/tmp/arm_lens.npy", arm_lens)
# np.save("../../data/tmp/angles.npy", angles)
dists = np.load("../../data/tmp/dists.npy")[:33]*22.7
arm_lens = np.load("../../data/tmp/arm_lens.npy")[:33]
angles = np.load("../../data/tmp/angles.npy")[:33]

fig, axes = plt.subplots(2, 2, figsize=(15,10))

axes[0,0].fill_between(xs, np.percentile(dists, 75, axis=1), np.percentile(dists, 90, axis=1), alpha=0.5, color="grey")
axes[0,0].fill_between(xs, np.percentile(dists, 50, axis=1), np.percentile(dists, 75, axis=1), alpha=0.7, color="grey")
axes[0,0].plot(xs, np.percentile(dists, 50, axis=1))
axes[0,0].set_title("A) Vertex Displacement")
axes[0,0].set_xlabel("# of smoothing iterations")
axes[0,0].set_ylabel("Distance moved ($\mu m$)")
axes[0,0].set_ylim(0, 200)
# axes[0,0].set_yscale("log")
axes[0,0].legend([f"{n}% upper quantile" for n in [90, 75]] + ["Median"], loc="upper left")

axes[0,1].plot(np.arange(500), barrier_vals)
axes[0,1].set_title("B) Barrier Criterion")
axes[0,1].set_xlabel("# of smoothing iterations")
axes[0,1].set_ylabel("Value of barrier criterion")
axes[0,1].set_yscale("log")

axes[1,0].plot(xs, np.any(arm_lens<3, axis=2).sum(axis=1)/arm_lens.shape[1]*100)
axes[1,0].plot(xs, np.any(arm_lens<2.5, axis=2).sum(axis=1)/arm_lens.shape[1]*100)
axes[1,0].set_title("C) Frequency of Short Vessels")
axes[1,0].set_xlabel("# of smoothing iterations")
axes[1,0].set_ylabel("% of vessels being short")
axes[1,0].legend(["$|e| < 3 \cdot R$", "$|e| < 2.5 \cdot R$"], loc="upper right")

angle_cnts = np.sum(angles < np.pi/10, axis=(1,2))/angles.shape[1]/2*100
axes[1,1].plot(xs, angle_cnts)
axes[1,1].set_title("D) Frequency of Colinear Cessels")
axes[1,1].set_xlabel("# of smoothing iterations")
axes[1,1].set_ylabel("% of vessels being colinear")
# axes[1,1].set_yscale("log")

fig.subplots_adjust(hspace=0.3, wspace=0.3)

plt.show()
