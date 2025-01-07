"""
Module for computing convolution surface grid using single threaded approach.

"""
from typing import Tuple, Callable
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm  # For printing progress bar
import numpy as np
import time

import data as DATA
from grid import Grid


def render_field(dx: float
                , iso_value: float
                , data: DATA.RenderData
                , kernel: Callable[[ArrayLike, ArrayLike, float, float], ArrayLike]
                , verbose: bool
                 ) -> Tuple[Grid, NDArray[np.float64]]:
    """
    Compute potential energy field of skeleton.

    :param dx:        The desired grid cell size to use.
    :param data:      A data structure that contains skeleton information.
    :param kernel:    A kernel function that we will use to for computing a convolution with the skeleton.
    :param verbose:   A boolean flag to toggle output on screen.
    :return:          A tuple with the computed grid, and a timings-array, where each entry holds the duration for
                      processing one edge in the skeleton.
    """
    start_time = time.time()
    values = np.full(data.dim, iso_value)
    end_time = time.time()
    if verbose:
        print("render_field(): Grid creation took ", end_time - start_time, " seconds")

    timings = np.zeros((data.K,), dtype=float)
    for idx in tqdm(range(data.K), disable=not verbose):
        timings[idx] = time.time()
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
        timings[idx] = time.time() - timings[idx]
    if verbose:
        print("render_field(...): Minimum edge time took ", np.min(timings), " seconds")
        print("render_field(...): Average edge time took ", np.mean(timings), " seconds")
        print("render_field(...): Maximum edge time took ", np.max(timings), " seconds")
    return Grid(values, dx, data.min), timings
