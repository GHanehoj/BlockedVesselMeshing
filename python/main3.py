import data as DATA
import pyvista as pv
import numpy as np
from tools.pyvista_plotting import *
from tools.mesh_util import *
from tools.contouring import graph_contour, contour
import convolution as CONV

# V = np.array([[0, 0, 0],
#               [3, 0, 0],
#               [3, 3, 0],
#               ])
# E = np.array([[0, 1],
#               [1,2],
#               ])

# R = np.array([1, 1, 0.1])

# V = np.array([[ 1.92043187e+07, -3.79652084e+07,  1.21580279e+08],
#               [ 1.92043877e+07, -3.79651390e+07,  1.21580472e+08]])

# E = np.array([[0, 1]])
# R = np.array([56.40347392, 73.59652608])
# dx = 43.38728763419388

V = np.array([[  0.6680756, -0.4986115,  3245.6411743],
              [  0.6699165, -0.4967573,  3245.6463426]])
E = np.array([[0, 1]])
R = np.array([0.0023165, 0.0030226])
dx = 0.001158243496698511

sz = 37459.556438578066
v, t = graph_contour(V, E, R, 4)
# grid = CONV.conv_surf(V*sz, E, R*sz*0.65, dx*sz)
# v, t = contour(grid)
# v /= sz

tris = np.concatenate((np.full((t.shape[0], 1), 3), t), axis=1)
grid = pv.PolyData(v, tris)

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(grid, 'lightgrey', lighting=True, show_edges=True, opacity=0.5)
# plotter.add_lines(V[E].reshape(-1,3), color="k")
# points = pv.PolyData(V)
# points['R'] = R/0.65
# spheres = points.glyph(geom=pv.Sphere(radius=1.0), scale="R")
# plotter.add_mesh(spheres, opacity=0.5, color="b")

plotter.camera_position = "xZ"
plotter.screenshot(f"./conv_test/{0}.png")