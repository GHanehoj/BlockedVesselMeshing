"""
Module for computing convolution surface grid using single threaded approach.

"""
from typing import Callable
from numpy.typing import ArrayLike, NDArray
import numpy as np
import data as DATA

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
        u, c = np.unique(E, return_counts=True)
        endsA = np.isin(E[:,0], u[c==1])
        endsB = np.isin(E[:,1], u[c==1])
        self.A[endsA] = self.A[endsA] - 0.3*self.U[endsA] * R[E[endsA,0]][:,np.newaxis]
        self.B[endsB] = self.B[endsB] - 0.3*self.U[endsB] * R[E[endsB,1]][:,np.newaxis]
        self.D = self.B - self.A  # Edge direction vector for all edges in E
        self.L = np.linalg.norm(self.D, axis=1)  # Edge lengths for all edges in E
        self.R = R  # vertex radius values
        self.P = 2.0 * np.max(R[E], axis=1)  # Padding magnitude around each edge in E, AABBs need to be large enough to cover the iso-surface of the skeleton.
        self.P[self.P < 5.0 * dx] = 5.0 * dx

        self.min_corner = np.min(np.stack((self.A, self.B), axis=1), axis=1) - self.P[:, np.newaxis]  # Minimum corner for AABB around each edge of E
        self.max_corner = np.max(np.stack((self.A, self.B), axis=1), axis=1) + self.P[:, np.newaxis]  # Maximum corner for AABB around each edge of E

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

class Grid:
    def __init__(self, values, dx, min):
        self.values = values
        self.dx = dx
        self.min = min
        self.dim = values.shape
        self.max = self.dim+self.min


def render_field(dx: float
                , iso_value: float
                , data: DATA.RenderData
                , kernel: Callable[[ArrayLike, ArrayLike, float, float], ArrayLike]
                 ) -> Grid:
    """
    Compute potential energy field of skeleton.

    :param dx:        The desired grid cell size to use.
    :param data:      A data structure that contains skeleton information.
    :param kernel:    A kernel function that we will use to for computing a convolution with the skeleton.
    :param verbose:   A boolean flag to toggle output on screen.
    :return:          A tuple with the computed grid, and a timings-array, where each entry holds the duration for
                      processing one edge in the skeleton.
    """
    values = np.full(data.dim, iso_value)

    for idx in range(data.K):
        a = data.A[idx]  # Starting point of edge
        l = data.L[idx]  # Length of the edge
        u = data.U[idx]  # Unit direction vector of edge
        r0 = data.R[data.E[idx, 0]]  # Radius of starting vertex
        r1 = data.R[data.E[idx, 1]]  # Radius of ending vertex
        #
        # Setup minimum/maximum range of node indices for all grid nodes that are inside the
        # AABB box that bounds the edge.
        #
        i_min = data.I_min[idx]
        i_max = data.I_max[idx]
        j_min = data.J_min[idx]
        j_max = data.J_max[idx]
        k_min = data.K_min[idx]
        k_max = data.K_max[idx]
        range_i = np.arange(i_min, i_max + 1)
        range_j = np.arange(j_min, j_max + 1)
        range_k = np.arange(k_min, k_max + 1)
        #
        # Convert the ranges of indices into 3D grids of indices.
        #
        i_3d, j_3d, k_3d = np.meshgrid(range_i, range_j, range_k, indexing='ij')
        #
        # Calculate p using vectorized operations; this will have dimension [3, I, J, K]
        # where I, J, and K are the length of range_i, range_j and range_k
        #
        p = np.array([i_3d * dx, j_3d * dx, k_3d * dx])
        #
        # Calculate alpha using vectorized operations, alpha is defined by $alpha = (p - a) * u$ where $a$ is the
        # starting point of the edge and $u$ is the unit direction vector.
        # Hence, alpha gives us the projection of p onto the edge.
        #
        alpha = np.sum((p - a[:, np.newaxis, np.newaxis, np.newaxis]) * u[:, np.newaxis, np.newaxis, np.newaxis],
                       axis=0)
        #
        # Calculate q using vectorized operations, q is the actual projection point of p onto the edge. That is
        #   q = a + alpha*u
        #
        q = a[:, np.newaxis, np.newaxis, np.newaxis] + alpha[np.newaxis, :, :, :] * u[:, np.newaxis, np.newaxis,
                                                                                    np.newaxis]
        #
        # Calculate beta using vectorized operations, beta is the distance from the projected
        # point q to the actual point p.
        #
        beta = np.linalg.norm(p - q, axis=0)
        #
        # Interpolate radius value between starting and ending vertex of the edge.
        #
        r = r0 + ((r1-r0)/l)*np.clip(alpha,0.0, l)
        #
        # Call the convolution surface function with flatten 1D array inputs
        #
        potential = kernel(alpha, beta, r, l)
        #
        # Add values back into the 3D grid
        #

        values[(i_min-data.min[0]):(i_max-data.min[0]+1), (j_min-data.min[1]):(j_max-data.min[1]+1), (k_min-data.min[2]):(k_max-data.min[2]+1)] -= potential
    return Grid(values, dx, data.min)
