import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
from tools.numpy_util import mk_mask
from rainbow.math.tetrahedron import compute_inscribed_sphere, compute_circumscribed_sphere
import meshio

### Segment mesh ###
class SegMesh:
    def __init__(self, nodes, segs):
        self.nodes = nodes
        self.segs = segs
    def calc_neighbours(self):
        L = np.zeros((len(self.nodes), len(self.nodes)))
        L[self.edges[:,0],self.edges[:,1]] = 1
        L[self.edges[:,1],self.edges[:,0]] = 1
        tots = np.sum(L, axis=1)
        tots[tots == 0] = 1
        L = L/tots[:, None]
        return L

    def clean(self):
        self.nodes, self.segs = clean_free_nodes(self.nodes, self.segs)

    def size(self):
        return (self.nodes.size + self.segs.size)*8/(10**6)

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

    def clean(self):
        self.nodes, self.tris = clean_free_nodes(self.nodes, self.tris)

    def size(self):
        return (self.nodes.size + self.tris.size)*8/(10**6)

    def save(self, file):
        meshio.Mesh(self.nodes, [("triangle", self.tris)]).write(file)

def merge_tri_meshes(meshes):
    v_szs = np.array([mesh.nodes.shape[0] for mesh in meshes])
    t_szs = np.array([mesh.tris.shape[0] for mesh in meshes])

    v_szs_cum = np.zeros_like(v_szs)
    v_szs_cum[1:] = np.cumsum(v_szs)[:-1]
    tOffsets = np.repeat(v_szs_cum, t_szs)

    nodes = np.concatenate([mesh.nodes for mesh in meshes])
    tris = np.concatenate([mesh.tris for mesh in meshes])+tOffsets[:, None]

    nodes, tris = merge_duplicate_nodes(nodes, tris)
    mesh = TriMesh(nodes, tris)
    mesh.clean()

    return mesh

def load_tri(file):
    msh = meshio.read(file)
    return TriMesh(msh.points, msh.cells_dict["triangle"])



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

    def clean(self):
        self.nodes, self.tets = clean_free_nodes(self.nodes, self.tets)

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


def load_tet(file):
    msh = meshio.read(file)
    return TetMesh(msh.points, msh.cells_dict["tetra"])

### Multi Tetrahedron Mesh ###
from typing import List
class MultiTetMesh:
    def __init__(self,
                 cluster_tet     : TetMesh,
                 child_tets      : List['MultiTetMesh'],
                 connector_tets  : List[TetMesh],
                 cluster_out_ends: List['EndSlice'],
                 child_in_ends   : List['EndSlice']):

        assert(len(child_tets) == len(connector_tets))
        k = len(child_tets)
        if k == 0:
            self.nodes = cluster_tet.nodes
            self.tets = cluster_tet.tets
            self.root_node_cnt = len(cluster_tet.nodes)
            self.root_tet_cnt = len(cluster_tet.tets)
            return

        n0 = len(cluster_tet.nodes)


        idx0 = n0
        nodes_list = [None]*(2*k+1)
        tets_list  = [None]*(2*k+1)
        nodes_list[0] = cluster_tet.nodes
        tets_list[0] = cluster_tet.tets

        for i in range(k):
            conn_remappings = np.full(len(connector_tets[i].nodes), -1, dtype=int)
            # cluster_eq = np.linalg.norm(cluster_out_ends[i].end.nodes[None, :, :] - cluster_tet.nodes[:, None, :], axis=2) < 1e-5
            # conn_cluster_eq = np.linalg.norm(cluster_out_ends[i].end.nodes[None, :, :] - connector_tets[i].nodes[:, None, :], axis=2) < 1e-5
            # cluster_remaps = np.argmax(cluster_eq, axis=0)
            # conn_cluster_mask = np.sum(conn_cluster_eq, axis=1) != 0
            # conn_remappings[conn_cluster_mask] = cluster_remaps[np.argmax(conn_cluster_eq[conn_cluster_mask], axis=1)]
            
            conn_eq = np.linalg.norm(cluster_tet.nodes[None, :, :] - connector_tets[i].nodes[:, None, :], axis=2) < 1e-5
            conn_cluster_mask = np.sum(conn_eq, axis=1) != 0
            conn_remappings[conn_cluster_mask] = np.argmax(conn_eq[conn_cluster_mask], axis=1)

            conn_node_cnt = (~conn_cluster_mask).sum()
            conn_remappings[~conn_cluster_mask] = np.arange(conn_node_cnt) + n0
            conn_nodes = connector_tets[i].nodes[~conn_cluster_mask]


            child_root_nodes = child_tets[i].nodes[:child_tets[i].root_node_cnt]
            child_root_tets = child_tets[i].tets[:child_tets[i].root_tet_cnt]
            child_other_nodes = child_tets[i].nodes[child_tets[i].root_node_cnt:]
            child_other_tets = child_tets[i].tets[child_tets[i].root_tet_cnt:]
            child_remappings = np.full(len(child_root_nodes), -1, dtype=int)
            # conn_child_eq = np.linalg.norm(child_in_ends[i].end.nodes[None, :, :] - connector_tets[i].nodes[:, None, :], axis=2) < 1e-5
            # child_eq = np.linalg.norm(child_in_ends[i].end.nodes[None, :, :] - child_root_nodes[:, None, :], axis=2) < 1e-5
            # conn_remaps = np.argmax(conn_child_eq, axis=0)
            # child_mask = np.sum(child_eq, axis=1) != 0
            # child_remappings[child_mask] = conn_remaps[np.argmax(child_eq[child_mask], axis=1)] + n0
            
            child_eq = np.linalg.norm(conn_nodes[None, :, :] - child_root_nodes[:, None, :], axis=2) < 1e-5
            child_mask = np.sum(child_eq, axis=1) != 0
            child_remappings[child_mask] = np.argmax(child_eq[child_mask], axis=1) + n0

            child_node_cnt = (~child_mask).sum()
            child_remappings[~child_mask] = np.arange(child_node_cnt) + conn_node_cnt + n0
            child_nodes = child_root_nodes[~child_mask]


            sub_mesh_nodes = np.concatenate((cluster_tet.nodes, conn_nodes, child_nodes))
            sub_mesh_tets = np.concatenate((cluster_tet.tets, conn_remappings[connector_tets[i].tets], child_remappings[child_root_tets]))
            sub_mesh = TetMesh(sub_mesh_nodes, sub_mesh_tets)
            _smooth_transition(sub_mesh, cluster_out_ends[i].flow_data)
            _smooth_transition(sub_mesh, child_in_ends[i].flow_data)
            nodes_list[0]     = sub_mesh.nodes[:n0]
            nodes_list[2*i+1] = sub_mesh.nodes[n0:n0+conn_node_cnt]
            nodes_list[2*i+2] = np.concatenate((sub_mesh.nodes[n0+conn_node_cnt:], child_other_nodes), axis=0)

            # nodes_list[0]     = cluster_tet.nodes
            # nodes_list[2*i+1] = conn_nodes
            # nodes_list[2*i+2] = np.concatenate((child_nodes, child_other_nodes), axis=0)

            conn_remappings[~conn_cluster_mask] += idx0-n0
            tets_list[2*i+1] = conn_remappings[connector_tets[i].tets]
            child_remappings += idx0-n0
            child_tet_root_mask = child_other_tets>=child_tets[i].root_node_cnt
            child_other_tets[~child_tet_root_mask] = child_remappings[child_other_tets[~child_tet_root_mask]]
            child_other_tets[child_tet_root_mask] += idx0+conn_node_cnt-child_mask.sum()
            tets_list[2*i+2] = np.concatenate((child_remappings[child_root_tets], child_other_tets), axis=0)

            idx0 += conn_node_cnt + child_node_cnt+len(child_other_nodes)


        self.nodes = np.concatenate(nodes_list, axis=0)
        self.tets  = np.concatenate(tets_list,  axis=0)
        self.root_node_cnt = len(cluster_tet.nodes)
        self.root_tet_cnt = len(cluster_tet.tets)

from tools.smoothing import laplacian_smoothing, taubin_smoothing
def _smooth_transition(tet_mesh, flow_data):
    p0 = tet_mesh.nodes-flow_data.point
    d = flow_data.dir*flow_data.radius

    mask = np.logical_and.reduce((np.dot(p0+d, flow_data.dir) > 0,
                                  np.dot(p0-d, flow_data.dir) < 0,
                                  np.linalg.norm(p0-np.dot(p0, flow_data.dir)[...,None]*flow_data.dir[None,:], axis=1) < 1.5*flow_data.radius))

    ## First, we smooth the surface.
    surface = tet_mesh.surface()
    surface_neigh = surface.calc_neighbours()
    laplacian_smoothing(tet_mesh.nodes, surface_neigh, ~mask, iter=10)

    # ## Then the volume
    # fixed_mask = np.logical_or(mask, mk_mask(np.unique(surface.tris), len(tet_mesh.nodes)))
    # volume_neigh = tet_mesh.calc_neighbours()
    # laplacian_smoothing(tet_mesh.nodes, volume_neigh, fixed_mask)

### General mesh functions ###

def clean_free_nodes(nodes, elems):
    """
    Removes mesh nodes that are not connected to anything.
    Keeps original arrays intact.

    :param nodes:             N x 3 array of vertex locations
    :param elems:             N x k array of mesh elements.
    :return:                  2 new arrays containing the cleaned mesh vertices and elements
    """
    mask = mk_mask(np.unique(elems), nodes.shape[0])
    offsets = np.cumsum(~mask)
    nodes = nodes[mask]
    elems = elems.copy()
    elems -= offsets[elems]
    return nodes, elems

import pyvista as pv
def merge_duplicate_nodes2(nodes, elems, tol=1e-5):
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

def merge_duplicate_nodes(nodes, elems, tol=1e-5):
    N, k = elems.shape
    pv_mesh = pv.PolyData(nodes, np.hstack((np.full((N,1), k), elems)))
    pv_mesh = pv_mesh.clean(tolerance=tol)
    nodes = pv_mesh.points
    elems = pv_mesh.faces.reshape(-1,k+1)[:,1:]
    return nodes, elems

def rad_ratio(tet):
    r_out = compute_circumscribed_sphere(*tet)[1]
    r_in = compute_inscribed_sphere(*tet)[1]
    return 3*r_in/r_out

def rad_ratios(tet_mesh):
    return np.array([rad_ratio(tet) for tet in tet_mesh.nodes[tet_mesh.tets]])

def flatness(tet):
    a = tet[:,0,:]
    b = tet[:,1,:]
    c = tet[:,2,:]
    d = tet[:,3,:]
    ab = a-b
    ac = a-c
    ad = a-d
    bc = b-c
    bd = b-d
    cd = c-d
    norm = lambda v: np.linalg.norm(v, axis=1)
    num = 80*np.abs(np.einsum("id,id->i", np.cross(ab, ac), ad))
    inv = (norm(ab)+norm(cd))*(norm(ac)+norm(bd))*(norm(ad)+norm(bc))
    mask = inv != 0
    flatness = np.empty(len(tet))
    flatness[~mask] = np.inf
    flatness[mask] = num[mask]/inv[mask]
    return flatness