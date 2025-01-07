"""
Grid class and subroutines to make life easier.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import tools.file as FILE
from tree import tree_conf_name
from branch import branch_name

class Grid:
    def __init__(self, values, dx, min):
        self.values = values
        self.dx = dx
        self.min = min
        self.dim = values.shape
        self.max = self.dim+self.min

        X = np.linspace(self.min[0]*self.dx, self.max[0]*self.dx, self.dim[0])
        Y = np.linspace(self.min[1]*self.dx, self.max[1]*self.dx, self.dim[1])
        Z = np.linspace(self.min[2]*self.dx, self.max[2]*self.dx, self.dim[2])

        self.sampler = RegularGridInterpolator((X,Y,Z), self.values, bounds_error=False, fill_value=1/32)



class BranchGrid:
    def __init__(self, values, dx, min, idx, tree_conf):
        self.values = values
        self.dx = dx
        self.min = min
        self.dim = values.shape
        self.max = self.dim+self.min
        self.idx = idx
        self.tree_conf = tree_conf

        X = np.linspace(self.min[0]*self.dx, self.max[0]*self.dx, self.dim[0])
        Y = np.linspace(self.min[1]*self.dx, self.max[1]*self.dx, self.dim[1])
        Z = np.linspace(self.min[2]*self.dx, self.max[2]*self.dx, self.dim[2])

        self.sampler = RegularGridInterpolator((X,Y,Z), self.values, bounds_error=False, fill_value=1/32)

def grid_file(idx, tree_conf, data_folder):
    return f"{data_folder}/grids/{tree_conf_name(tree_conf)}/{branch_name(idx)}.npz"

def save_np_grid(np_grid, data_folder):
    file = grid_file(np_grid.idx, np_grid.tree_conf, data_folder)
    FILE.create_folders_if_not_exist(file)
    np.savez(file, grid=np_grid.grid, dx=np_grid.dx, min=np_grid.min)

# def load_np_grid(idx, tree_conf, data_folder):
#     file = grid_file(idx, tree_conf, data_folder)
#     with np.load(file) as data:
#         return Grid(data['grid'], data['dx'], data['min'], idx, tree_conf)
