import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import os
import re
import json
from plyfile import PlyData
from tools.numpy_util import mk_mask

### Skeleton format ###
def load_skeleton_data(folder: str):
    """
    Load skeleton data from numpy arrays.

    :param folder:                 The folder containing 3 graph data files.
    :param verbose:                Boolean flag for toggling text output.
    :return: A triplet of vertices (V), edges (E) and radius (R) arrays
    """
    V = np.load(folder+'/vertex_array.npy')
    E = np.load(folder+'/edge_array.npy')
    R = np.load(folder+'/vertex_radius_array.npy')

    return V, E, R

def save_skeleton_data(V,E,R, folder: str):
    np.save(folder+'/vertex_array.npy', V)
    np.save(folder+'/edge_array.npy', E)
    np.save(folder+'/vertex_radius_array.npy', R)


### VesselGen format ###
def load_vesselgen_data(folder: str):
    V = np.load(folder+"_coords.npy")
    E = np.load(folder+"_connections.npy")
    Re = np.load(folder+"_radii.npy")
    r_acc = np.zeros(len(V))
    n_acc = np.zeros(len(V))
    for e,r in zip(E,Re):
        r_acc[e[0]] += r
        r_acc[e[1]] += r
        n_acc[e[0]] += 1
        n_acc[e[1]] += 1
    R = r_acc/n_acc

    return V, E, R

### VesselGraph format ###
def load_vesselgraph_data(vertices_file: str, edges_file: str):
    vertices_data = np.genfromtxt(vertices_file, delimiter=";", skip_header=1)
    V = vertices_data[:, 1:4]
    edges_data = np.genfromtxt(edges_file, delimiter=";", skip_header=1)
    E = edges_data[:, 1:3].astype(np.int64)
    Re = edges_data[:, 10]
    r_acc = np.zeros(len(V))
    n_acc = np.zeros(len(V))
    for e,r in zip(E,Re):
        r_acc[e[0]] += r
        r_acc[e[1]] += r
        n_acc[e[0]] += 1
        n_acc[e[1]] += 1
    n_acc[n_acc == 0] = 1
    R = r_acc/n_acc

    return V, E, R

### 3DSlicer Segmented Curves ###
def load_segmented_curves(folder: str):
    files = os.listdir(folder)
    curves = dict()
    for file in files:
        m = re.search('^.*\((\d+)\).mrk.json$', file)
        if m is None: continue
        with open(folder+file) as f:
            data = json.load(f)
            points = np.array([p["position"] for p in data["markups"][0]["controlPoints"]])
            radii = np.array([measure["controlPointValues"] for measure in data["markups"][0]["measurements"] if measure["name"] == "Radius"]).flatten()
            curves[int(m.group(1))] = (points, radii)

    V = np.empty((0,3))
    E = np.empty((0,2), dtype=np.int64)
    R = np.empty((0))
    for i in range(len(curves)):
        points, radii = curves[i]
        i0 = len(V)
        V = np.vstack((V, points))
        i1 = len(V)-1
        i = np.arange(i0, i1)
        edges = np.vstack((i, i+1)).T
        E = np.vstack((E, edges))
        R = np.hstack((R, radii))

    V, E, R = merge_vertices(V, E, R)

    return V, E, R


### VTK format ###
def load_vtk_data(folder):
    V = np.genfromtxt(folder+"/V.vtk", delimiter=" ")
    E = np.genfromtxt(folder+"/E.vtk", delimiter=" ", dtype=np.int64)[:,1:]
    R = np.genfromtxt(folder+"/R.vtk", delimiter=" ")

    _, idx, inv = np.unique(V, axis=0, return_index=True, return_inverse=True)
    mask = mk_mask(np.unique(idx), len(V))
    remappings = idx[inv]
    offsets = np.cumsum(~mask)

    V = V[mask]
    E = remappings[E]
    E -= offsets[E]
    
    r_acc = np.zeros(len(idx))
    n_acc = np.zeros(len(idx))
    for i, ii in enumerate(inv):
        r_acc[ii] += R[i]
        n_acc[ii] += 1
    R = r_acc/n_acc

    return V, E, R

### AdTree format ###
def load_adtree(file, rdp_eps):
    ply = PlyData.read(file)
    V = np.array([[v["x"], v["y"], v["z"]] for v in ply.elements[0]])
    E = np.array([e[0] for e in ply.elements[1]])
    R = np.array([v["radius"] for v in ply.elements[0]])

    V, E, R = merge_vertices(V, E, R)

    return V, E, R


def merge_vertices(V, E, R = None):
    _, idx, inv = np.unique(V, axis=0, return_index=True, return_inverse=True)
    mask = mk_mask(np.unique(idx), len(V))
    remappings = idx[inv]
    offsets = np.cumsum(~mask)

    V = V[mask]
    E = remappings[E]
    E -= offsets[E]
    E = E[E[:,0] != E[:,1]]

    if R is not None:
        R = R[mask]
        return V, E, R
    else:
        return V, E