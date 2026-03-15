import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np

def create_folders_if_not_exist(filepath: str) -> None:
    # Extract directory path from the given file path
    directory = os.path.dirname(filepath)

    # Create directories recursively if they don't exist
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_npy(file, npy):
    create_folders_if_not_exist(file)
    np.save(file, npy)


def load_skeleton_data(folder: str):
    """
    Load skeleton data from numpy arrays.

    :param folder:                 The folder containing 3 graph data files.
    :return: A triplet of vertices (V), edges (E) and radius (R) arrays
    """
    V = np.load(folder+'/vertex_array.npy')
    E = np.load(folder+'/edge_array.npy')
    R = np.load(folder+'/vertex_radius_array.npy')

    return V, E, R

def save_skeleton_data(V,E,R, folder: str):
    create_folders_if_not_exist(folder)
    np.save(folder+'/vertex_array.npy', V)
    np.save(folder+'/edge_array.npy', E)
    np.save(folder+'/vertex_radius_array.npy', R)
