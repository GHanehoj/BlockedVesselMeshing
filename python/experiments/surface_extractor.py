import sys
import os
sys.path.append(os.path.abspath('../'))
from tools.mesh_util import *
path = "/media/data/data/meshes/meshes/lung_3"

def surface(tet):
    tri_idxs = [[0,1,2], [1,2,3], [0,2,3], [0,1,3]]

    all_tris = np.sort(tet.tets[:, tri_idxs], axis=2).reshape(-1, 3)
    print("sorted")
    unique_tris, counts = np.unique(all_tris, axis=0, return_counts=True)
    print("unique_found")
    tris = unique_tris[counts == 1]

    return TriMesh(tet.nodes, tris)


def perspective_fov(fov, aspect_ratio, near_plane, far_plane):
	num = 1.0 / np.tan(fov / 2.0)
	num9 = num / aspect_ratio
	return np.array([
		[num9, 0.0, 0.0, 0.0],
		[0.0, num, 0.0, 0.0],
		[0.0, 0.0, -(far_plane+near_plane) / (far_plane - near_plane), -2*(near_plane * far_plane) / (far_plane - near_plane)],
		[0.0, 0.0, -1, 0.0]
	])


def view_cull(tet, view, proj, scale):
    p_hom = np.hstack((tet.nodes,np.ones((len(tet.nodes), 1))))
    p_cam = view@p_hom.T
    p_norm = np.vstack((p_cam[:3]/scale, p_cam[3]))
    p_clip = (proj@p_norm)
    p = (p_clip[:3].T / p_clip[3][:,None])

    cull_mask = np.logical_or.reduce((p[:,0] < -1, p[:,0] > 1, p[:,1] < -1, p[:,1] > 1, p[:,2] < -1, p[:,2] > 1))
    offsets = np.cumsum(cull_mask)
    tet_mask = np.any(cull_mask[tet.tets], axis=1)
    nodes = tet.nodes[~cull_mask]
    tets = tet.tets[~tet_mask]
    tets -= offsets[tets]

    return TetMesh(nodes, tets)


# tet = load_tet(f"{path}.mesh")
# surf = surface(tet)
# surf.save(f"{path}_surf.mesh")


tet = load_tet(f"{path}.mesh")

view = np.array([-0.571226894855499,0.82079154253006,6.27478584647179e-08,83.1867294311523,-0.494486629962921,-0.344136148691177,0.798156201839447,-323.962554931641,0.655120491981506,0.455928266048431,0.602451026439667,-150.457702636719,0.0,0.0,0.0,1.0]).reshape(4,4)
proj = perspective_fov(np.pi/4, 2/3, 0.005, 20.0)
scale = 289.0601806640625
culled = view_cull(tet, view, proj, scale)

culled.save(f"{path}_culled.mesh")


