import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
from tools.numpy_util import mk_mask
from collections import namedtuple
import meshio

### Segment mesh ###
SegMesh = namedtuple("SegMesh", ["nodes", "segs"])

def calc_neighbours_seg(mesh):
    L = np.zeros((len(mesh.nodes), len(mesh.nodes)))
    L[mesh.edges[:,0],mesh.edges[:,1]] = 1
    L[mesh.edges[:,1],mesh.edges[:,0]] = 1
    tots = np.sum(L, axis=1)
    tots[tots == 0] = 1
    L = L/tots[:, None]
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
class TriMesh:
    def __init__(self, nodes, tris):
        self.nodes = nodes
        self.tris = tris

    def calc_neighbours(self):
        L = np.zeros((len(self.nodes), len(self.nodes)))
        edge_idxs = [[0,1], [1,2], [2,0]]
        all_edges = self.tris[:,edge_idxs].reshape(-1, 2)
        L[all_edges[:,0],all_edges[:,1]] = 1
        L[all_edges[:,1],all_edges[:,0]] = 1
        tots = np.sum(L, axis=1)
        tots[tots == 0] = 1
        L = L/tots[:, None]
        return L

    def edge(self):
        edge_idxs = [[0,1], [1,2], [2,0]]

        all_edges = np.sort(self.tris[:, edge_idxs], axis=2).reshape(-1, 2)
        unique_edges, counts = np.unique(all_edges, axis=0, return_counts=True)
        edges = unique_edges[counts != 2]

        return SegMesh(self.nodes, edges)

    def get_poly_edges(self):
        edge_idxs = [[0,1], [1,2], [2,0]]

        all_edges = np.sort(self.tris[:, edge_idxs], axis=2).reshape(-1, 2)
        unique_edges, counts = np.unique(all_edges, axis=0, return_counts=True)
        return unique_edges[counts > 2]

    def collapse_edges(self, edges):
        edges = np.sort(edges, axis=1)
        remove_mask = mk_mask(edges[:, 1], len(self.nodes))
        offsets = np.cumsum(remove_mask)

        remappings = np.arange(len(self.nodes))
        remappings -= offsets
        remappings[edges[:, 1]] = edges[:, 0]

        edge_idxs = [[0,1], [1,2], [2,0]]

        tri_edges = np.sort(self.tris[:, edge_idxs], axis=2)
        tri_mask = np.any(np.all(tri_edges[:,:,None, :] == edges[None, None, :, :], axis=3), axis=(1,2))

        self.nodes = self.nodes[~remove_mask]
        self.tris = self.tris[~tri_mask]
        self.tris = remappings[self.tris]
        # return SegMesh(self.nodes, edges)

    def size(self):
        return (self.nodes.size + self.tris.size)*8/(10**6)

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


### Tetrahedron mesh ###
class TetMesh:
    def __init__(self, nodes, tets):
        self.nodes = nodes
        self.tets = tets

    def calc_neighbours(self):
        L = np.zeros((len(self.nodes), len(self.nodes)))
        edge_idxs = [[0,1], [1,2], [2,0], [0,3], [1,3], [2,3]]
        all_edges = self.tets[:,edge_idxs].reshape(-1, 2)
        L[all_edges[:,0],all_edges[:,1]] = 1
        L[all_edges[:,1],all_edges[:,0]] = 1
        tots = np.sum(L, axis=1)
        tots[tots == 0] = 1
        L = L/tots[:, None]
        return L
        
    def surface(self):
        tri_idxs = [[0,1,2], [1,2,3], [0,2,3], [0,1,3]]

        all_tris = np.sort(self.tets[:, tri_idxs], axis=2).reshape(-1, 3)
        unique_tris, counts = np.unique(all_tris, axis=0, return_counts=True)
        tris = unique_tris[counts == 1]

        return TriMesh(self.nodes, tris)

    def collapse_edges(self, edges):
        edges = np.sort(edges, axis=1)
        remove_mask = mk_mask(edges[:, 1], len(self.nodes))
        offsets = np.cumsum(remove_mask)

        remappings = np.arange(len(self.nodes))
        remappings -= offsets
        remappings[edges[:, 1]] = edges[:, 0]

        edge_idxs = [[0,1], [1,2], [2,0], [0,3], [1,3], [2,3]]

        tet_edges = np.sort(self.tets[:, edge_idxs], axis=2)
        tet_mask = np.any(np.all(tet_edges[:,:,None, :] == edges[None, None, :, :], axis=3), axis=(1,2))

        self.nodes = self.nodes[~remove_mask]
        self.tets = self.tets[~tet_mask]
        self.tets = remappings[self.tets]
        # return SegMesh(self.nodes, edges)

    def size(self):
        return (self.nodes.size + self.tets.size)*8/(10**6)

    def save(self, file):
        meshio.Mesh(self.nodes, [("tetra", self.tets)]).write(file)

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


### Multi Tetrahedron Mesh ###
class MultiTetMesh:
    def __init__(self):
        self.nodes = np.empty((0,3))                 # N x 3     array of vertex locations
        self.tets = np.empty((0,4), dtype=int)       # T x 4     array of tetrahedron vertex indices
        self.aabbs = np.empty((0,2,3))               # K x 2 x 3 list of AABB for each mesh
        self.node_idxs = np.empty((0,2), dtype=int)  # K x 2     list of vertex slice for each mesh
        self.tet_idxs = np.empty((0,2), dtype=int)   # K x 2     list of tetrahedron slice for each mesh

    def append_mesh(self, tet: TetMesh):
        """
        Assumes the new mesh is clean (no superfluous nodes), and merges
        it with the existing efficiently.

        :param self:              Existing MultiTetMesh, to be extended
        :param tet:               Additional TetMesh to add
        """
        ## give some padding eps!
        tet_aabb = np.vstack((tet.nodes.min(axis=0), tet.nodes.max(axis=0)))

        mins_below_max = np.all(self.aabbs[:, 0, :] <= tet_aabb[None, 1, :], axis=-1)
        maxs_above_min = np.all(self.aabbs[:, 1, :] >= tet_aabb[None, 0, :], axis=-1)
        aabb_overlap = np.logical_and(mins_below_max, maxs_above_min)

        rgs = self.node_idxs[aabb_overlap]
        remappings = np.full(len(tet.nodes), -1, dtype=int)
        for a, b in rgs:
            eq = np.linalg.norm(self.nodes[None, a:b, :] - tet.nodes[:, None, :], axis=2) < 1e-5
            mask = np.sum(eq, axis=1) != 0
            remappings[mask] = np.argmax(eq[mask], axis=1) + a


        new_mask = remappings == -1
        n0 = len(self.nodes)
        new_nodes = tet.nodes[new_mask]

        remappings[new_mask] = np.arange(len(new_nodes))+n0
        t0 = len(self.tets)
        new_tets = remappings[tet.tets]

        self.nodes     = np.concatenate((self.nodes, new_nodes), axis=0)
        self.tets      = np.concatenate((self.tets, new_tets), axis=0)
        self.aabbs     = np.concatenate((self.aabbs, [tet_aabb]), axis=0)
        self.node_idxs = np.concatenate((self.node_idxs, [[n0, n0+len(new_nodes)]]), axis=0)
        self.tet_idxs  = np.concatenate((self.tet_idxs, [[t0, t0+len(new_tets)]]), axis=0)

        return len(self.aabbs)-1

    def get_sub_mesh(self, ids):
        tets = np.concatenate([self.tets[a:b] for a,b in self.tet_idxs[ids]])
        node_mask = mk_mask(np.unique(tets), len(self.nodes))
        offsets = np.cumsum(~node_mask)
        tets -= offsets[tets]
        nodes = self.nodes[node_mask]
        return TetMesh(nodes, tets)

    def write_back(self, sub_mesh, ids):
        tets = np.concatenate([self.tets[a:b] for a,b in self.tet_idxs[ids]])
        node_mask = mk_mask(np.unique(tets), len(self.nodes))
        self.nodes[node_mask] = sub_mesh.nodes

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