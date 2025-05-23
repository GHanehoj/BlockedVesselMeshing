import sys
import os
sys.path.append(os.path.abspath('../'))
import preprocessing.tree_smoothing as SMOOTH
import load as LOAD
import tools.file as FILE

def save(V, E, R, mask, changes, file):
    trees_folder = os.path.dirname(__file__)+file
    FILE.save_skeleton_data(V, E, R,
                            vertex_array_file  = trees_folder+'/vertex_array.npy',
                            edge_array_file    = trees_folder+'/edge_array.npy',
                            vertex_radius_file = trees_folder+'/vertex_radius_array.npy')

    FILE.save_npy(trees_folder+"/mask.npy", mask)
    FILE.save_npy(trees_folder+"/changes.npy", changes)


def preprocess(V, E, R):
    # V, E, R = SPLIT.ensure_cardinality(V, E, R)
    regSave = lambda V, E, R, chg, i: save(V, E, R, [], chg, f"/../../data/input/brain/VesselGen/result_3/preprocessed/reg{i}")
    V, E, R, changes = SMOOTH.smooth(V, E, R, iter=101, lap_lr=0.04, rep_lr=0.02, bar_lr=25, save_fn=regSave)
    # V, E, R, changes = SMOOTH.smooth(V, E, R, iter=101, lap_lr=0.2, rep_lr=0.15, bar_lr=100, save_fn=regSave)
    return V, E, R, [], changes

if __name__ == '__main__':
    tree_folder = f"../../data/input/brain/VesselGen/result_3/skeleton"
    V, E, R = LOAD.load_skeleton_data(tree_folder)

    V, E, R, mask, changes = preprocess(V, E, R)

    save(V, E, R, mask, changes, "/../../data/input/brain/VesselGen/result_3/preprocessed/regDone")
