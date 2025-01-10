import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
from collections import namedtuple

### Segment mesh ###
SegMesh = namedtuple("SegMesh", ["nodes", "segs"])

def calc_neigbours_seg(mesh):
    L = np.zeros((len(mesh.nodes), len(mesh.nodes)))
    for e0, e1 in mesh.segs:
        L[e0,e1] = 1
        L[e1,e0] = 1
    tots = np.sum(L, axis=1)
    mask = tots != 0
    L[mask] = L[mask]/tots[mask, None]
    return L

def merge_seg_meshes(meshes):
    v_szs = np.array([mesh.nodes.shape[0] for mesh in meshes])
    s_szs = np.array([mesh.segs.shape[0] for mesh in meshes])

    v_szs_cum = np.zeros_like(v_szs)
    v_szs_cum[1:] = np.cumsum(v_szs)[:-1]
    sOffsets = np.repeat(v_szs_cum, s_szs)

    nodes = np.concatenate([mesh.nodes for mesh in meshes])
    segs = np.concatenate([mesh.segs for mesh in meshes])+sOffsets[:, None]

    nodes, segs = merge_duplicate_nodes(nodes, segs)

    return SegMesh(nodes, segs)

def clean_seg_mesh(mesh):
    nodes, segs = clean_free_nodes(mesh.nodes, mesh.segs)
    return SegMesh(nodes, segs)

def seg_mesh_size(mesh):
    return (mesh.nodes.size + mesh.segs.size)*8/(10**6)


### Triangle mesh ###
TriMesh = namedtuple("TriMesh", ["nodes", "tris"])

def calc_neigbours_tri(mesh):
    L = np.zeros((len(mesh.nodes), len(mesh.nodes)))
    edge_idxs = [[0,1], [1,2], [2,0]]
    for t in mesh.tris:
        for e0, e1 in t[edge_idxs]:
            L[e0,e1] = 1
            L[e1,e0] = 1
    tots = np.sum(L, axis=1)
    mask = tots != 0
    L[mask] = L[mask]/tots[mask, None]
    return L

def merge_tri_meshes(meshes):
    v_szs = np.array([mesh.nodes.shape[0] for mesh in meshes])
    t_szs = np.array([mesh.tris.shape[0] for mesh in meshes])

    v_szs_cum = np.zeros_like(v_szs)
    v_szs_cum[1:] = np.cumsum(v_szs)[:-1]
    tOffsets = np.repeat(v_szs_cum, t_szs)

    nodes = np.concatenate([mesh.nodes for mesh in meshes])
    tris = np.concatenate([mesh.tris for mesh in meshes])+tOffsets[:, None]

    raw_mesh = TriMesh(nodes, tris)

    return clean_tri_mesh(raw_mesh)

def clean_tri_mesh(mesh):
    nodes, tris = clean_free_nodes(mesh.nodes, mesh.tris)
    return TriMesh(nodes, tris)

def surface_edge(mesh):
    edge_idxs = [[0,1], [1,2], [2,0]]

    all_edges = np.sort(mesh.tris[:, edge_idxs], axis=2).reshape(-1, 2)
    ns = np.sum(np.all(all_edges[:, None, :] == all_edges[None, :, :], axis=2), axis=1)
    edges = np.unique(all_edges.reshape(-1, 3, 2)[np.where(ns.reshape(-1, 3) != 2)], axis=0)

    return SegMesh(mesh.nodes, edges)

def tri_mesh_size(mesh):
    return (mesh.nodes.size + mesh.tris.size)*8/(10**6)


### Tetrahedron mesh ###
TetMesh = namedtuple("TetMesh", ["nodes", "tets"])

def calc_neigbours_tet(mesh):
    L = np.zeros((len(mesh.nodes), len(mesh.nodes)))
    edge_idxs = [[0,1], [1,2], [2,0], [0,3], [1,3], [2,3]]
    for t in mesh.tets:
        for e0, e1 in t[edge_idxs]:
            L[e0,e1] = 1
            L[e1,e0] = 1
    tots = np.sum(L, axis=1)
    mask = tots != 0
    L[mask] = L[mask]/tots[mask, None]
    return L

def merge_tet_meshes(meshes):
    v_szs = np.array([mesh.nodes.shape[0] for mesh in meshes])
    t_szs = np.array([mesh.tets.shape[0] for mesh in meshes])

    v_szs_cum = np.zeros_like(v_szs)
    v_szs_cum[1:] = np.cumsum(v_szs)[:-1]
    tOffsets = np.repeat(v_szs_cum, t_szs)

    nodes = np.concatenate([mesh.nodes for mesh in meshes])
    tets = np.concatenate([mesh.tets for mesh in meshes])+tOffsets[:, None]

    nodes, tets = merge_duplicate_nodes(nodes, tets)

    return TetMesh(nodes, tets)

def clean_tet_mesh(mesh):
    nodes, tets = clean_free_nodes(mesh.nodes, mesh.tets)
    return TetMesh(nodes, tets)

def volume_surface(mesh):
    tri_idxs = [[0,1,2], [1,2,3], [0,2,3], [0,1,3]]

    all_tris = np.sort(mesh.tets[:, tri_idxs], axis=2).reshape(-1, 3)
    ns = np.sum(np.all(all_tris[:, None, :] == all_tris[None, :, :], axis=2), axis=1)
    tris = np.unique(all_tris.reshape(-1, 4, 3)[np.where(ns.reshape(-1, 4) == 1)], axis=0)

    return TriMesh(mesh.nodes, tris)

def tet_mesh_size(mesh):
    return (mesh.nodes.size + mesh.tets.size)*8/(10**6)


### General mesh functions ###

def clean_free_nodes(nodes, elems):
    """
    Removes mesh nodes that are not connected to anything.
    Keeps original arrays intact.

    :param nodes:             N x 3 array of vertex locations
    :param elems:             N x k array of mesh elements.
    :return:                  2 new arrays containing the cleaned mesh vertices and elements
    """
    mask = np.any(np.arange(nodes.shape[0])[:, None, None] == elems[None, :, :], axis=(1,2))
    offsets = np.cumsum(~mask)
    nodes = nodes[mask]
    elems = elems.copy()
    elems -= offsets[elems]
    return nodes, elems

def merge_duplicate_nodes(nodes, elems, tol=1e-5):
    """
    Merges nodes that are positionally identical, combining their
    connectivity information. Keeps original arrays intact.

    :param nodes:             N x 3 array of vertex locations
    :param elems:             N x k array of mesh elements.
    :param tol:               Distance tolerance between vertices considered identical
    :return:                  2 new arrays containing the merged mesh vertices and elements
    """
    eq = np.triu(np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2) < tol)
    mask = np.sum(eq, axis=0) == 1
    remappings = np.argmax(eq, axis=0)
    offsets = np.cumsum(~mask)
    nodes = nodes[mask]
    elems = remappings[elems]
    elems -= offsets[elems]
    return nodes, elems