import numpy as np
import tree as TREE
import formats as FORMATS

data_path = "../../data/"

### VesselGen: Brain ###
V, E, R = FORMATS.load_vesselgen_data(data_path+"input/brain/raw/result_3/test_1_result_12")
E = np.vstack((E, [[290,758]])) # Connect two segments in this dataset.

root, _ = TREE.make_tree_unordered2(V, E, R, True)
TREE.simplify_edges(root, 0.08)
TREE.merge_groupings(root, 0.7)
V, E, R = TREE.make_arrays(TREE.get_nodes(root))

FORMATS.save_skeleton_data(V, E, R, data_path+"input/brain/skeleton/")


### 3DSlicer: Lung ###
V, E, R = FORMATS.load_segmented_curves(data_path+"input/lung/raw/segmented_curves/")

root, _ = TREE.make_tree_unordered2(V, E, R)
TREE.simplify_edges(root, 0.08)
TREE.prune_tiny_leaves(root, 10e-3)
TREE.widen_leaves(root, 0.2)
V, E, R = TREE.make_arrays(TREE.get_nodes(root))

FORMATS.save_skeleton_data(V, E, R, data_path+"input/lung/skeleton/")


### VTK: Liver ###
V, E, R = FORMATS.load_vtk_data(data_path+"input/liver/raw/")

root, _ = TREE.make_tree_unordered2(V, E, R)
TREE.prune_tiny_offshoots(root)
TREE.simplify_edges(root, 0.08)
TREE.merge_groupings(root, 0.7)
TREE.limit_radius_growth(root, 2)
V, E, R = TREE.make_arrays(TREE.get_nodes(root))

FORMATS.save_skeleton_data(V, E, R, data_path+"input/liver/skeleton/")


### AdTree: Tree ###
V, E, R = FORMATS.load_adtree(data_path+"input/tree/raw/ahn3_delft.ply")

root, _ = TREE.make_tree_unordered2(V, E, R)
TREE.prune_zero_radius(root)
TREE.prune_tiny_offshoots(root)
TREE.simplify_edges(root, 0.08)
TREE.merge_groupings(root, 0.7)
V, E, R = TREE.make_arrays(TREE.get_nodes(root))

FORMATS.save_skeleton_data(V, E, R, data_path+"input/tree/ahn/skeleton/")
