import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import torch
import gc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_CAP = 10.0
def cap_diff(V_diff):
    V_dists = torch.linalg.vector_norm(V_diff, dim=1)
    mask = V_dists > _CAP
    V_diff[mask] *= (_CAP/V_dists[mask, None])
    return V_diff

def compute_neighbour_matrix(V, E):
    """
    Computes a neighbour matrix where the i'th entry contains
    the indices of the neighbours of vertex i. Since all nodes
    are of cardinality <4, this can be stored in (Nx3).
    Missing neighbours are represented as -1.
    """
    max_N = np.max(np.unique(E, return_counts=True)[1])
    neighbours = np.full((len(V), max_N), -1, dtype=np.int32)
    for e in E:
        n0 = neighbours[e[0]]
        n0[np.sum(n0 != -1)] = e[1]
        n1 = neighbours[e[1]]
        n1[np.sum(n1 != -1)] = e[0]
    return neighbours

def normalize(v):
    return v / torch.linalg.vector_norm(v, dim=-1)[...,None]

def angle_between(v1, v2):
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    return torch.arccos(torch.clip(torch.einsum("...k,...k->...",v1_u, v2_u), -1.0, 1.0))

def laplacian_smooth(V, neighbours, lr):
    internal_mask = np.sum(neighbours != -1, axis=1) > 1

    centers = V[neighbours[internal_mask]]
    arms = centers-V[internal_mask, None, :]
    mids = (centers[:,[0,1,2],:] + centers[:,[1,2,0],:])/2

    angs = angle_between(arms[:,[0,1,2],:], arms[:,[1,2,0],:])

    # angs = angle_between(arms[:,:,None,:], arms[:,None,:,:])
    weights = torch.pi/torch.maximum(angs, torch.tensor(0.01))
    weights /= torch.sum(weights,dim=1)[:,None]

    D = torch.sum(weights[:,:,None]*mids, dim=1)-V[internal_mask]

    V[internal_mask] += cap_diff(lr * D)

def repel_smooth(V, R, neighbours, lr):
    centers = V[neighbours]
    arms = centers-V[:, None, :]

    lens = torch.linalg.vector_norm(arms, dim=2)
    lens[neighbours == -1] = torch.inf
    repels = 3/2*B3(2/3 * lens / (R[neighbours]*4))

    V -= cap_diff(lr * torch.nansum(repels[:,:,None] * arms, dim=1))

def calc_g_spaces(V, E, R):
    def aabb(e):
        radii = np.mean(R[E[e]], axis=1)
        padding = np.stack((-radii, radii), axis=1)
        aabb = np.sort(V[E[e]], axis=-2)
        return aabb + padding[:, :, None]

    e_space = np.arange(len(E))
    aabbs = aabb(e_space)

    mins_below_maxs = np.all(aabbs[:, None, 0, :] <= aabbs[None, :, 1, :], axis=2)
    gc.collect()
    maxs_above_mins = np.all(aabbs[:, None, 1, :] >= aabbs[None, :, 0, :], axis=2)
    gc.collect()
    aabb_overlap = np.logical_and(mins_below_maxs, maxs_above_mins)
    gc.collect()

    edges = E[e_space]
    not_neighbour = np.all(edges[:, None, :, None] != edges[None, :, None, :], axis=(2,3))
    gc.collect()

    g_space = np.logical_and(aabb_overlap, not_neighbour)
    gc.collect()

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
        return barrier_kernel(d(e, g, s, t), torch.tensor(4.0)*h(e, g, s, t))

    def psi(e, K, g_space):
        if g_space.size == 0: return 0
        k_space = torch.linspace(0, 1, K, device=device)
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
        V_new = V.clone().detach()
    
    V += V_prev-V_new
    V += cap_diff(V_new-V_prev)
    return psi_tot.cpu().detach().numpy()


def smooth(V, E, R, iter, lap_lr, rep_lr, bar_lr, save_fn = None):
    V_torch = torch.tensor(V, device=device, requires_grad=True)
    E_torch = torch.tensor(E, device=device)
    R_torch = torch.tensor(R, device=device)
    with torch.no_grad():
        # g_spaces = calc_g_spaces(V_torch.cpu().detach().numpy(), E, R)
        # g_spaces = np.zeros((len(E),len(E)))
        neighbours = compute_neighbour_matrix(V_torch, E_torch)
        changes = []
        for i in range(iter):
            V_prev = V_torch.clone().detach()
            laplacian_smooth(V_torch, neighbours, lap_lr)

            V_lap = V_torch.clone().detach()
            repel_smooth(V_torch, R_torch, neighbours, rep_lr)

            # V_rep = V_torch.clone().detach()
            # barrier_val = barrier_optim(V_torch, E_torch, R_torch, g_spaces, bar_lr)

            max_change_lap = torch.max(torch.linalg.vector_norm(V_prev-V_lap, dim=1)).cpu().detach().numpy()
            max_change_rep = torch.max(torch.linalg.vector_norm(V_lap-V_torch, dim=1)).cpu().detach().numpy()
            # max_change_bar = torch.max(torch.linalg.vector_norm(V_torch-V_torch, dim=1)).cpu().detach().numpy()
            mean_change_lap = torch.mean(torch.linalg.vector_norm(V_prev-V_lap, dim=1)).cpu().detach().numpy()
            mean_change_rep = torch.mean(torch.linalg.vector_norm(V_lap-V_torch, dim=1)).cpu().detach().numpy()
            # mean_change_bar = torch.mean(torch.linalg.vector_norm(V_torch-V_torch, dim=1)).cpu().detach().numpy()
            changes.append([max_change_lap, max_change_rep, mean_change_lap, mean_change_rep])
            print(f"iteration {i} --- {max_change_lap:.4f}, {max_change_rep:.4f}, {mean_change_lap:.4f}, {mean_change_rep:.4f}")

            if save_fn and (i < 5 or (i < 100 and i % 5 == 0) or i % 50 == 0):
                save_fn(V_torch.cpu().detach().numpy(), E, R, changes, i)
            # if i % 50 == 49 and i != iter-1:
            #     g_spaces = np.array([])
            #     gc.collect()
            #     g_spaces = calc_g_spaces(V_torch.cpu().detach().numpy(), E, R)

    return V_torch.cpu().detach().numpy(), E, R, changes
