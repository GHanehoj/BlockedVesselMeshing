import sys
import os
sys.path.append(os.path.abspath('../'))
from typing import List
import numpy as np

def create_folders_if_not_exist(filepath: str) -> None:
    # Extract directory path from the given file path
    directory = os.path.dirname(filepath)

    # Create directories recursively if they don't exist
    if not os.path.exists(directory):
        os.makedirs(directory)


def is_valid_file_extension(filepath: str, valid_extensions: List[str]) -> bool:
    # Extract file extension from the given file path
    _, extension = os.path.splitext(filepath)

    # Check if the file extension is in the list of valid extensions
    return extension.lower() in valid_extensions


def file_exists(filename):
    return os.path.exists(filename)


def save_npy(file, npy):
    create_folders_if_not_exist(file)
    np.save(file, npy)

def save_skeleton_data(V, E, R, vertex_array_file: str, edge_array_file: str, vertex_radius_file: str):
    create_folders_if_not_exist(vertex_array_file)
    create_folders_if_not_exist(edge_array_file)
    create_folders_if_not_exist(vertex_radius_file)
    np.save(vertex_array_file, V)
    np.save(edge_array_file, E)
    np.save(vertex_radius_file, R)
