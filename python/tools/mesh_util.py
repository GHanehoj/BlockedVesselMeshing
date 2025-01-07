import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
from collections import namedtuple

SegMesh = namedtuple("SegMesh", ["nodes", "segs"])
TriMesh = namedtuple("TriMesh", ["nodes", "tris"])
TetMesh = namedtuple("TetMesh", ["nodes", "tets"])

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

def merge_tet_meshes(meshes):
    v_szs = np.array([mesh.nodes.shape[0] for mesh in meshes])
    t_szs = np.array([mesh.tets.shape[0] for mesh in meshes])

    v_szs_cum = np.zeros_like(v_szs)
    v_szs_cum[1:] = np.cumsum(v_szs)[:-1]
    tOffsets = np.repeat(v_szs_cum, t_szs)

    nodes = np.concatenate([mesh.nodes for mesh in meshes])
    tets = np.concatenate([mesh.tets for mesh in meshes])+tOffsets[:, None]

    return TetMesh(nodes, tets)


def clean_tri_mesh(mesh):
    nodes, tris = mesh.nodes, mesh.tris
    nodes, tris = clean_free_nodes(nodes, tris)
    nodes, tris = merge_duplicate_nodes(nodes, tris)
    return TriMesh(nodes, tris)

def clean_tet_mesh(mesh):
    nodes, tets = mesh.nodes, mesh.tets
    nodes, tets = clean_free_nodes(nodes, tets)
    nodes, tets = merge_duplicate_nodes(nodes, tets)
    return TetMesh(nodes, tets)

def tet_mesh_size(mesh):
    return (3*mesh.nodes.shape[0] + 4*mesh.tets.shape[0])*8/(10**6)
def tri_mesh_size(mesh):
    return (3*mesh.nodes.shape[0] + 3*mesh.tris.shape[0])*8/(10**6)

def clean_free_nodes(nodes, elems):
    mask = np.any(np.arange(nodes.shape[0])[:, None, None] == elems[None, :, :], axis=(1,2))
    offsets = np.cumsum(~mask)
    nodes = nodes[mask]
    elems -= offsets[elems]
    return nodes, elems

def merge_duplicate_nodes(nodes, elems):
    eq = np.triu(np.all(nodes[:, None, :] == nodes[None, :, :], axis=2))
    mask = np.sum(eq, axis=0) == 1
    remappings = np.argmax(eq, axis=0)
    offsets = np.cumsum(~mask)
    nodes = nodes[mask]
    elems = remappings[elems]
    elems -= offsets[elems]
    return nodes, elems