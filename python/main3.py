import convolution as CONV
from tools.pyvista_plotting import add_graph, add_tri_mesh
from tools.mesh_util import TriMesh
import numpy as np
import pyvista as pv

np.seterr(divide="raise", invalid="raise")

# V = np.array([[  20.74163055, -211.53378296,  500.25067139],
#               [  16.65366123, -204.85019919,  488.14837862]])
# E = np.array([[0,1],
#               ])
# R = np.array([7.20845468, 7.65938599])
# dx = 4.805636453065841


# V = np.array([[   1.4369278,  -179.97175598,  443.09976196],
#               [   6.73252002, -188.6297305,   458.77718084],
#               [   1.09354871, -167.88709037,  428.86500765]])
# E = np.array([[0, 1],
#               [0, 2]])
# R = np.array([9.33789701, 8.75375654, 9.5944984 ])
# dx = 6.03055118299488

# V = np.array([[   0.63452274, -151.73242188,  1409.83612061],
#               [   1.04234335, -166.08500158,  1426.74229346],
#               [  17.79930901, -138.61503142,  1400.85526071],
#               [ -18.29409145, -150.5849441,   1396.41759793]])
V = np.array([[2.0, 0, 100],
              [1, 0, 100]])
E = np.array([[0, 1],
            #   [0, 2],
            #   [0, 3]
              ])
# R = np.array([9.93752074, 9.6327633,  7.51466556, 7.88824343])
R = np.array([0.4, 0.5])
# dx = 5.695866939065866
dx = 0.1

grid = CONV.conv_surf_SCALIS(V, E, R, dx)
v, t = CONV.contour(grid)

plotter = pv.Plotter()
add_tri_mesh(plotter, TriMesh(v, t))
add_graph(plotter, V, E, R)
plotter.show()