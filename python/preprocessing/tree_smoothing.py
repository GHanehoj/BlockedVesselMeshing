import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import torch
import gc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_CAP = 1.0
def cap_diff(V_diff):
    V_dists = torch.linalg.vector_norm(V_diff, dim=1)
    mask = V_dists > _CAP
    V_diff[mask] *= (_CAP/V_dists[mask, None])
    return V_diff

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

def angle_between(v1, v2):
    v1_u = v1/torch.linalg.vector_norm(v1, dim=-1)[...,None]
    v2_u = v2/torch.linalg.vector_norm(v2, dim=-1)[...,None]
    return torch.arccos(torch.clip(torch.einsum("ijk,ijk->ij",v1_u, v2_u), -1.0, 1.0))

def laplacian_smooth(V, neighbors, lr):
    internal_mask = np.sum(neighbors != -1, axis=1) == 3

    centers = V[neighbors[internal_mask]]
    D = torch.mean(centers, dim=1) - V[internal_mask]

    arms = centers-V[internal_mask, None, :]
    idxs = np.array([[0,1], [1,2], [2,0]])
    angs = angle_between(arms[:, idxs[:,0]], arms[:, idxs[:,1]])
    ang_weight = 1/torch.maximum(torch.min(angs, axis=1)[0]/torch.pi, torch.tensor(0.005))

    V[internal_mask] += cap_diff(lr * ang_weight[:, None] * D)

def repel_smooth(V, R, neighbors, lr):
    internal_mask = np.sum(neighbors != -1, axis=1) == 3

    centers = V[neighbors[internal_mask]]
    arms = centers-V[internal_mask, None, :]

    lens = torch.linalg.vector_norm(arms, dim=2)
    repels = 3/2*B3_np(2 * lens / (R[internal_mask, None]*4))

    V[internal_mask] -= cap_diff(lr * torch.sum(repels[:,:,None] * arms, dim=1))

# def combined_smooth(V, R, neighbors, lr1, lr2):
#     internal_mask = np.sum(neighbors != -1, axis=1) == 3

#     centers = V[neighbors[internal_mask]]
#     D = torch.mean(centers, dim=1) - V[internal_mask]

#     arms = centers-V[internal_mask, None, :]
#     idxs = np.array([[0,1], [1,2], [2,0]])
#     angs = angle_between(arms[:, idxs[:,0]], arms[:, idxs[:,1]])
#     ang_weight = 1/torch.maximum(torch.min(angs, axis=1)[0]/torch.pi, torch.tensor(0.01))

#     V[internal_mask] += cap_diff(lr1 * ang_weight[:, None] * D)


#     lens = torch.linalg.vector_norm(arms, axis=2)
#     repels = 3/2*B3_np(2 * lens / (R[internal_mask, None]*4))

#     V[internal_mask] -= cap_diff(lr2 * torch.sum(repels[:,:,None] * arms, dim=1))

def B3_np(u):
    return torch.where(u < 1, 2/3 - u*u + 1/2 * u*u*u,
           torch.where(u < 2, 1/6 * (2-u)*(2-u)*(2-u),
                           0.0))

def calc_g_spaces(V, E, R):
    def aabb(e):
        radii = np.mean(R[E[e]], axis=1)
        padding = np.stack((-radii, radii), axis=1)
        aabb = np.sort(V[E[e]], axis=-2)
        return aabb + padding[:, :, None]

    e_space = np.arange(len(E))
    aabbs = aabb(e_space)

    print("A")
    mins_below_maxs = np.all(aabbs[:, None, 0, :] <= aabbs[None, :, 1, :], axis=2)
    gc.collect()
    print("B")
    maxs_above_mins = np.all(aabbs[:, None, 1, :] >= aabbs[None, :, 0, :], axis=2)
    gc.collect()
    print("C")
    aabb_overlap = np.logical_and(mins_below_maxs, maxs_above_mins)
    gc.collect()
    print("D")

    edges = E[e_space]
    not_neighbor = np.all(edges[:, None, :, None] != edges[None, :, None, :], axis=(2,3))
    gc.collect()
    print("E")

    g_space = np.logical_and(aabb_overlap, not_neighbor)
    gc.collect()
    print("F")

    return g_space

def barrier_kernel(d, h):
    return 3*B3(2 * d / h)/(2 * d * d)
def B3(u):
    return torch.where(u < 1, 2/3 - u*u + 1/2 * u*u*u,
           torch.where(u < 2, 1/6 * (2-u)*(2-u)*(2-u),
                              0*u))

def barrier_optim(V, E, R, g_spaces, lr):
    def d(e, g, s, t):
        va, vb = V[E[e]][:, None, None, None, :]
        vc, vd = torch.einsum("kij->ikj", V[E[g]])[:, :, None, None, :]
        pe = va + s[None, :, None, None]*(vb-va)
        pg = vc + t[None, None, :, None]*(vd-vc)
        return torch.linalg.vector_norm(pe - pg.detach(), dim=3)
    def h(e, g, s, t):
        ra, rb = R[E[e]][:, None, None, None]
        rc, rd = torch.einsum("ki->ik", R[E[g]])[:, :, None, None]
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
    
    with torch.enable_grad():
        optimizer = torch.optim.SGD([V], lr=lr)
        optimizer.zero_grad()
        psi_tot = 0
        e_space = np.arange(len(E))
        for e in e_space:
            psi_tot += psi(e, 10, e_space[g_spaces[e]])
        psi_tot.backward()

        V_prev = V.clone().detach()
        optimizer.step()
    V += cap_diff(V-V_prev)
    return psi_tot.cpu().detach().numpy()


def smooth(V, E, R, iter=201, lap_lr=0.003, rep_lr=0.05, bar_lr=0.01, save_fn = None):
    V_torch = torch.tensor(V, device=device, requires_grad=True)
    E_torch = torch.tensor(E, device=device)
    R_torch = torch.tensor(R, device=device)
    with torch.no_grad():
        g_spaces = calc_g_spaces(V_torch.cpu().detach().numpy(), E, R)
        # g_spaces = np.zeros((len(E),len(E)))
        neighbors = compute_neigbor_matrix(V_torch, E_torch)
        changes = []
        for i in range(iter):
            V_prev = V_torch.clone().detach()
            laplacian_smooth(V_torch, neighbors, lap_lr)

            V_lap = V_torch.clone().detach()
            repel_smooth(V_torch, R_torch, neighbors, rep_lr)

            V_rep = V_torch.clone().detach()
            barrier_val = barrier_optim(V_torch, E_torch, R_torch, g_spaces, bar_lr)

            max_change_lap = torch.max(torch.linalg.vector_norm(V_prev-V_lap, dim=1)).cpu().detach().numpy()
            max_change_rep = torch.max(torch.linalg.vector_norm(V_lap-V_rep, dim=1)).cpu().detach().numpy()
            max_change_bar = torch.max(torch.linalg.vector_norm(V_rep-V_torch, dim=1)).cpu().detach().numpy()
            mean_change_lap = torch.mean(torch.linalg.vector_norm(V_prev-V_lap, dim=1)).cpu().detach().numpy()
            mean_change_rep = torch.mean(torch.linalg.vector_norm(V_lap-V_rep, dim=1)).cpu().detach().numpy()
            mean_change_bar = torch.mean(torch.linalg.vector_norm(V_rep-V_torch, dim=1)).cpu().detach().numpy()
            changes.append([max_change_lap, max_change_rep, max_change_bar, mean_change_lap, mean_change_rep, mean_change_bar, barrier_val])
            print(f"iteration {i} --- {max_change_lap:.4f}, {max_change_rep:.4f}, {max_change_bar:.4f}, {mean_change_lap:.4f}, {mean_change_rep:.4f}, {mean_change_bar:.4f}, {barrier_val:.4f}")

            if save_fn and (i < 5 or (i < 100 and i % 5 == 0) or i % 50 == 0):
                save_fn(V_torch.cpu().detach().numpy(), E, R, changes, i)
            if i % 50 == 49 and i != iter-1:
                g_spaces = np.array([])
                gc.collect()
                g_spaces = calc_g_spaces(V_torch.cpu().detach().numpy(), E, R)

    return V_torch.cpu().detach().numpy(), E, R, changes
