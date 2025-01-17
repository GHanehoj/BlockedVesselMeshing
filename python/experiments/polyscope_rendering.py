import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import polyscope as ps
import meshio
import data as DATA


# def show_vessel_tree(V, E, R, show_tree = True, show_nodes = False, ):
#     min_corner = np.min(V, axis=0)
#     max_corner = np.max(V, axis=0)
#     center = np.mean(V, axis=0)

#     ps.set_up_dir("y_up")
#     ps.set_navigation_style("free")
#     ps.set_screenshot_extension(".png")

#     ps.init()
#     ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
#     ps.set_ground_plane_height_factor(min_corner[1], is_relative=False)  # adjust the plane height
#     ps.set_shadow_darkness(0.1)  # lighter shadows

#     if show_tree:
#         ps.register_curve_network(
#             "tree",
#             V,
#             E,
#             enabled=True,
#             material='candy',
#             radius=0.001,
#             color=(0.1, 0.1, 0.9),
#             transparency=1.
#         )
#         # edge_colors = np.zeros((len(R), 3))
#         # intensity = np.log(1.0 + R / np.max(R))
#         # intensity /= np.max(intensity)
#         # # Large radius is more red, low radius is more blue
#         # edge_colors[:, 0] = R / np.max(R)
#         # edge_colors[:, 2] = 1.0 - intensity
#         # ps_net.add_color_quantity("radia", edge_colors, defined_on='edges', enabled=True)

#         # ps.look_at(
#         #     camera_location=(27000, 5000, 9500),
#         #     target=(6000, 6000, 10000)
#         # )
#         # ps.screenshot(
#         #     filename="kidney_view_1.png",
#         #     transparent_bg=True
#         # )

#         # ps.look_at(
#         #     camera_location=(24300, 9600, 9700),
#         #     target=(6000, 5000, 10200)
#         # )
#         # ps.screenshot(
#         #     filename="kidney_view_2.png",
#         #     transparent_bg=True
#         # )
#         # ps.look_at(
#         #     camera_location=(12600, 24300, 9700),
#         #     target=(7400, 5000, 10200)
#         # )
#         # ps.screenshot(
#         #     filename="kidney_view_3.png",
#         #     transparent_bg=True
#         # )

#         # ps.look_at(
#         #     camera_location=(12000, 8000, 9700+8500),
#         #     target=(7400, 5000, 10200+8500)
#         # )
#         # ps.screenshot(
#         #     filename="cortex_view_2.png",
#         #     transparent_bg=True
#         # )

#     # if show_nodes:
#         # ps_cloud = ps.register_point_cloud("bc_new", V[valencies>3], color=(0.4, 0.9, 0.9), radius=0.003)
#         # node_colors = np.zeros((len(V), 3))
#         # color1 = [0.1, 0.1, 0.1]
#         # color2 = [0.7, 0.7, 0.7]
#         # color3 = [0.0, 1.0, 0.0]
#         # color4 = [0.0, 0.0, 1.0]
#         # color5 = [1.0, 1.0, 0.0]
#         # color6 = [1.0, 0.0, 1.0]
#         # color7 = [0.0, 1.0, 1.0]
#         # color8 = [1.0, 0.0, 0.0]
#         # node_colors[:, 0] = np.array(valencies)/ np.max(valencies)
#         # node_colors[valencies==1] = color1
#         # node_colors[valencies==2] = color2
#         # node_colors[valencies==3] = color3
#         # node_colors[valencies==4] = color4
#         # node_colors[valencies==5] = color5
#         # node_colors[valencies==6] = color6
#         # node_colors[valencies==7] = color7
#         # node_colors[valencies==8] = color8
#         # ps_cloud.add_color_quantity("valencies", node_colors[valencies>3], enabled=True)

#     ps.show()


# V, E, R = DATA.load_skeleton_data("../../data/trees/_reg200")
# show_vessel_tree(V, E, R)

# from tools.mesh_util import TetMesh

ps.set_up_dir("x_up")
# ps.set_navigation_style("free")
# ps.set_screenshot_extension(".png")

ps.init(backend='openGL_mock')
ps.set_window_size(800, 600)
ps.set_ground_plane_mode("shadow_only")  # set +Z as up direction
# ps.set_ground_plane_height_factor(min_corner[1], is_relative=False)  # adjust the plane height
# ps.set_shadow_darkness(0.1)  # lighter shadows

# mesh = meshio.read("out.mesh")
mesh = meshio.read("out.mesh")
print("loaded")
verts = mesh.points
tets = mesh.cells_dict["tetra"]

ps_vol = ps.register_volume_mesh("test volume mesh", verts, tets=tets)

n_vert = verts.shape[0]
n_cell = tets.shape[0]

# Add a scalar function on vertices
# data_vert = np.random.rand(n_vert)
# ps_vol.add_scalar_quantity("my vertex val", data_vert)

# you can also access the structure by name
# ps.get_volume_mesh("test volume mesh").add_scalar_quantity("my vertex val", data_vert)

# Add a scalar function on cells (with some options set)
# data_cell = np.random.rand(n_cell)
# ps_vol.add_scalar_quantity("my cell val", data_cell, defined_on='cells',
#                            vminmax=(-3., 3.), cmap='blues')

# ps_plane = ps.add_scene_slice_plane()
# ps_plane.set_draw_plane(True) # render the semi-transparent gridded plane
# ps_plane.set_draw_widget(True)
center = np.array([375.61790647, 251.04540943, 557.70237451])
ps.look_at(
    camera_location=center-np.array([0, 500, 0]),
    target=center
)
ps.screenshot(
    filename="cortex_view_1.png",
    transparent_bg=False
)
# Show the GUI
# ps.show()