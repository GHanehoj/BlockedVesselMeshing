"""
Data handling module for convolution surfaces.
"""
import numpy as np
from numpy.typing import NDArray

class RenderData:
    """
    This class supports pre-processing of skeleton data provided as an edge-indexed data
    structure with radius attributes.

    It essentially computes a flat view of the data where each "row" corresponds to one stick that
    needs to be "rendered" into a grid data structure.
    """

    def __init__(self, V: NDArray[np.float64], E: NDArray[np.int64], R: NDArray[np.float64], dx: float) -> None:
        """
        Initialization.

        :param V:    The vertex coordinates of the skeleton. Each row holds x, y and z coordinate of the vertex.
        :param E:    The edges of the skeleton. Each row holds the vertex indices that define the start
                     and end points of the edge.
        :param R:    Vertex radius attributes, one per vertex in the skeleton.
        :param dx:   The desired minimum grid spacing in the grid structure that will be filled.
        """
        self.K = len(E)  # Number of edges in the skeleton.
        self.E = E
        self.A = V[E[:, 0]]  # Start point of each edge in E
        self.B = V[E[:, 1]]  # End point of each edge in E
        self.D = self.B - self.A  # Edge direction vector for all edges in E
        self.L = np.linalg.norm(self.D, axis=1)  # Edge lengths for all edges in E
        self.U = self.D / self.L[:, np.newaxis]  # Unit direction vector for all edges in E
        self.R = R  # vertex radius values
        self.P = 2.0 * np.max(R[E], axis=1)  # Padding magnitude around each edge in E, AABBs need to be large enough to cover the iso-surface of the skeleton.
        self.P[self.P < 5.0 * dx] = 5.0 * dx

        self.min_corner = np.min(V[E], axis=1) - self.P[:, np.newaxis]  # Minimum corner for AABB around each edge of E
        self.max_corner = np.max(V[E], axis=1) + self.P[:, np.newaxis]  # Maximum corner for AABB around each edge of E

        # Calculate 'i_min', 'i_max', 'j_min', 'j_max', 'k_min', 'k_max' for all elements in E
        self.I_min = (self.min_corner[:, 0] / dx).astype(int)
        self.I_max = (self.max_corner[:, 0] / dx).astype(int)
        self.J_min = (self.min_corner[:, 1] / dx).astype(int)
        self.J_max = (self.max_corner[:, 1] / dx).astype(int)
        self.K_min = (self.min_corner[:, 2] / dx).astype(int)
        self.K_max = (self.max_corner[:, 2] / dx).astype(int)

        self.min = np.array([np.min(self.I_min), np.min(self.J_min), np.min(self.K_min)])
        self.max = np.array([np.max(self.I_max), np.max(self.J_max), np.max(self.K_max)])
        self.dim = self.max - self.min + 1

def load_skeleton_data(folder: str):
    """
    Load skeleton data from numpy arrays.

    :param folder:                 The folder containing 3 graph data files.
    :param verbose:                Boolean flag for toggling text output.
    :return: A triplet of vertices (V), edges (E) and radius (R) arrays
    """
    V = np.load(folder+'/vertex_array.npy')  # We bring V and R arrays into the same units.
    E = np.load(folder+'/edge_array.npy')
    R = np.load(folder+'/vertex_radius_array.npy')

    return V, E, R


# def select_grid_resolution(V: NDArray[np.float64]
#                            , E: NDArray[np.int64]
#                            , R: NDArray[np.float64]
#                            , number_of_cells: int = 4096
#                            , number_of_faces: int = 12
#                            , verbose: bool = False
#                            ) -> float:
#     """
#     This function selects the grid cell spacing (aka resolution) parameter.

#     :param V:                  The vertices of the skeleton.
#     :param E:                  The edge-index array of the skeleton.
#     :param R:                  The vertex-radius array of the skeleton.
#     :param number_of_cells:    Maximum number of cells along any coordinate axis.
#     :param number_of_faces:    Minimum number of faces around or along an edge.
#     :param verbose:            Boolean flag to toggle text output.
#     :return:                   The grid cell spacing to use for the skeleton.
#     """
#     L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)  # Edge lengths of skeleton
#     min_V = np.min(V, axis=0)  # Minimum corner point of skeleton
#     max_V = np.max(V, axis=0)  # Maximum corner point of skeleton
#     min_L = np.min(L)
#     min_R = np.min(R)

#     dx_grid = np.min((max_V - min_V) / number_of_cells)  # Trying to add a bound on grid resolution
#     dx_radius = 2.0 * np.pi * min_R / number_of_faces    # Voxel size to capture smallest radius information
#     dx_length = min_L / number_of_faces                  # Voxel size to capture the smallest length information
#     dx = max(dx_grid, min(dx_radius, dx_length))         # Clamp estimated voxel size by bounded voxel size.
#     res = np.ceil((max_V - min_V) / dx)                  # Effective resolution needed
#     if verbose:
#         print("select_grid_resolution(...): Smallest allowed voxel size:", dx_grid)
#         print("select_grid_resolution(...): Radius bound on voxel size:", dx_radius)
#         print("select_grid_resolution(...): Length bound on voxel size:", dx_length)
#         print("select_grid_resolution(...): Actual voxel size used:", dx)
#         print("select_grid_resolution(...): Effective needed grid resolution: ", res)
#         print("select_grid_resolution(...): Needed memory foot print of dense float grid:",
#               np.ceil((res[0] * res[1] * res[2] * 6) / 1024 / 1024 / 1024), "GB")
#     return dx


def subdivide_skeleton_data(V: NDArray[np.float64]
                            , E: NDArray[np.int64]
                            , R: NDArray[np.float64]
                            , max_subdivisions: int = 10
                            , verbose: bool = False
                            ):
    """
    Subdivides the edges of a skeleton, ensuring no edge is longer than its diameter,
    using precomputed average radii for each vertex, and updates the radius of each edge.

    :param V: Vertex array, each row holds x, y, and z coordinate of a vertex.
    :param E: Edge array, each row holds two vertex indices that correspond to the end-points of a skeleton edge.
    :param R: Radius array, holds the radius for each vertex.
    :param max_subdivisions: Maximum number of allowed subdivisions to perform.
    :param verbose: Boolean flag for toggling text output.
    :return: Updated vertex, edge, and edge radius arrays along with the updated average radius per vertex.
    """
    for n in range(max_subdivisions):
        if verbose:
            print("subdivide_skeleton(...): Running subdivision batch number:", n)

        L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)  # Compute the edge lengths
        # Determine edges to be subdivided based on the diameter at their vertices
        indices = np.where(L > 2.0 * np.max(R[E]))[0]
        if len(indices) == 0:
            if verbose:
                print("subdivide_skeleton(...): Done")
            break
        if verbose:
            print('subdivide_skeleton(...): Subdividing ', len(indices), ' edges')

        mid_vertices = (V[E[indices, 0]] + V[E[indices, 1]]) / 2.0
        new_vertex_indices = np.arange(len(V), len(V) + len(indices))

        # Update vertices and their average radii
        V = np.vstack([V, mid_vertices])
        new_R = (R[E[indices, 0]] + R[E[indices, 1]]) / 2.0
        R = np.hstack([R, new_R])

        # Prepare and update edges and their radii
        new_edges_1 = np.vstack([E[indices, 0], new_vertex_indices]).T
        new_edges_2 = np.vstack([new_vertex_indices, E[indices, 1]]).T
        E = np.vstack([E, new_edges_1, new_edges_2])

        # Remove subdivided original edges
        E = np.delete(E, indices, axis=0)

    return V, E, R
