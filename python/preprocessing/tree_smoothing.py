import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import torch
import gc
# from tqdm import tqdm
def tqdm(x): return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_CAP = 1.0
def cap_change(V_new, V):
    V_diff = V_new-V
    V_dists = np.linalg.norm(V_diff, axis=1)
    V_capped = np.where(V_dists[:, None] < _CAP, V + V_diff, V + V_diff/V_dists[:, None]*_CAP)
    return V_capped


def compute_neigbor_matrix(V, E):
    """
    Computes a neighbor matrix where the i'th entry contains
    the indices of the neighbors of vertex i. Since all nodes
    are of cardinality <4, this can be stored in (Nx3).
    Missing neighbors are represented as -1.
    """

    neighbors = np.empty((len(V), 3), dtype=np.int32)
    neighbors[:] = -1
    for e in E:
        n0 = neighbors[e[0]]
        n0[np.sum(n0 != -1)] = e[1]
        n1 = neighbors[e[1]]
        n1[np.sum(n1 != -1)] = e[0]
    return neighbors

def laplacian_smooth(V, neighbors, lr):
    centers = V[neighbors]
    centers[neighbors == -1] = np.nan
    C = np.nanmean(centers, axis=1)

    internal_mask = np.sum(neighbors != -1, axis=1) == 3

    V_new = V.copy()
    V_new[internal_mask] = V[internal_mask] + lr * (C[internal_mask] - V[internal_mask])

    return cap_change(V_new)

def repel_smooth(V, R, neighbors, lr):
    arms = V[neighbors]-V[:, None, :]

    arm_lens = np.linalg.norm(arms, axis=2)
    arm_lens[neighbors == -1] = 10e10

    repels = 3/2*B3_np(2/3 * arm_lens / R[:, None])

    V_new = V - lr * np.sum(repels[:,:,None] * arms, axis=1)

    return cap_change(V_new)

def B3_np(u):
    return np.where(not np.isinf and u < 1, 2/3 - u*u + 1/2 * u*u*u,
           np.where(u < 2, 1/6 * (2-u)*(2-u)*(2-u),
                           0))

def calc_g_spaces(V, E, R):
    def aabb(e):
        radii = np.mean(R[E[e]], axis=1)
        pad3d = np.repeat(radii[:, None], 3, axis=1)
        padding = np.transpose(np.array([-pad3d, pad3d]), (1,0,2))
        aabb = np.sort(V[E[e]], axis=-2)
        return aabb + padding

    e_space = np.arange(len(E))
    aabbs = aabb(e_space)

    mins_below_maxs = np.all(aabbs[:, None, 0, :] <= aabbs[None, :, 1, :], axis=2)
    gc.collect()
    maxs_above_mins = np.all(aabbs[:, None, 1, :] >= aabbs[None, :, 0, :], axis=2)
    gc.collect()
    aabb_overlap = np.logical_and(mins_below_maxs, maxs_above_mins)
    gc.collect()

    edges = E[e_space]
    gc.collect()
    not_neighbor = np.all(edges[:, None, :, None] != edges[None, :, None, :], axis=(2,3))

    g_space = np.logical_and(aabb_overlap, not_neighbor)
    gc.collect()

    return g_space

def barrier_kernel(d, h):
    return 3*B3(2 * d / h)/(2 * d * d)
def B3(u):
    return torch.where(u < 1, 2/3 - u*u + 1/2 * u*u*u,
           torch.where(u < 2, 1/6 * (2-u)*(2-u)*(2-u),
                              0*u))

def barrier_optim(V, E, R, g_spaces, lr):
    V_torch = torch.tensor(V, requires_grad=True, device=device)
    E_torch = torch.tensor(E, device=device)
    R_torch = torch.tensor(R, device=device)

    def d(e, g, s, t):
        va, vb = V_torch[E_torch[e]][:, None, None, None, :]
        vc, vd = torch.einsum("kij->ikj", V_torch[E_torch[g]])[:, :, None, None, :]
        pe = va + s[None, :, None, None]*(vb-va)
        pg = vc + t[None, None, :, None]*(vd-vc)
        return torch.linalg.vector_norm(pe - pg.detach(), dim=3)
    def h(e, g, s, t):
        ra, rb = R_torch[E_torch[e]][:, None, None, None]
        rc, rd = torch.einsum("ki->ik", R_torch[E_torch[g]])[:, :, None, None]
        re = ra + s[None, :, None]*(rb-ra)
        rg = rc + t[None, None, :]*(rd-rc)
        return re + rg
    def F(e, g, s, t):
        return barrier_kernel(d(e, g, s, t), h(e, g, s, t))

    def psi(e, K, g_space):
        if g_space.size == 0: return 0
        k_space = torch.linspace(1/(2*K), 1-1/(2*K), K, device=device)
        g_space = torch.tensor(g_space, device=device)
        vals = F(e, g_space, k_space, k_space)
        return torch.sum(vals)/(K*K)

    optimizer = torch.optim.SGD([V_torch], lr=lr)
    optimizer.zero_grad()
    psi_tot = 0
    e_space = np.arange(len(E))
    for e in tqdm(e_space):
        psi_tot += psi(e, 10, e_space[g_spaces[e]])
    psi_tot.backward()
    optimizer.step()
    V_new = V_torch.cpu().detach().numpy()
    return cap_change(V_new), psi_tot.cpu().detach().numpy()


def smooth(V, E, R, iter=200, lap_lr=0.002, rep_lr=0.01, bar_lr=0.001, save_fn = None):
    g_spaces = calc_g_spaces(V, E, R)
    neighbors = compute_neigbor_matrix(V, E)
    changes = []
    for i in range(iter):
        V_prev = V.copy()
        V = laplacian_smooth(V, neighbors, lap_lr)

        V_lap = V.copy()
        V = repel_smooth(V, R, neighbors, rep_lr)

        V_rep = V.copy()
        V, barrier_val = barrier_optim(V, E, R, g_spaces, bar_lr)

        max_change_lap = np.max(np.linalg.norm(V_prev-V_lap, axis=1))
        max_change_rep = np.max(np.linalg.norm(V_lap-V_rep, axis=1))
        max_change_bar = np.max(np.linalg.norm(V_rep-V, axis=1))
        mean_change_lap = np.mean(np.linalg.norm(V_prev-V_lap, axis=1))
        mean_change_rep = np.mean(np.linalg.norm(V_lap-V_rep, axis=1))
        mean_change_bar = np.mean(np.linalg.norm(V_rep-V, axis=1))
        changes.append([max_change_lap, max_change_rep, max_change_bar, mean_change_lap, mean_change_rep, mean_change_bar, barrier_val])
        print(f"iteration {i} --- {max_change_lap:.4f}, {max_change_rep:.4f}, {max_change_bar:.4f}, {mean_change_lap:.4f}, {mean_change_rep:.4f}, {mean_change_bar:.4f}, {barrier_val:.4f}")

        if save_fn and (i < 5 or (i < 100 and i % 5 == 0) or i % 50 == 0):
            save_fn(V, E, R, changes, i)
        if i % 50 == 49:
            g_spaces = np.array([])
            gc.collect()
            g_spaces = calc_g_spaces(V, E, R)

    return V, E, R, changes
