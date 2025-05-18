"""
Data handling module for convolution surfaces.
"""
import numpy as np
from numpy.typing import NDArray
from tools.numpy_util import mk_mask

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
