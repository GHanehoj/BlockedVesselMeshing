import sys
import os
sys.path.append(os.path.abspath('../'))
import preprocessing.vertex_splitter as SPLIT
import preprocessing.tree_smoothing as SMOOTH
import preprocessing.arterial_generation as ARTE
import data as DATA
import tools.file as FILE

def save(V, E, R, mask, changes, id):
    trees_folder = os.path.dirname(__file__)+"/../../data/trees/"+id
    FILE.save_skeleton_data(V, E, R,
                            vertex_array_file  = trees_folder+'/vertex_array.npy',
                            edge_array_file    = trees_folder+'/edge_array.npy',
                            vertex_radius_file = trees_folder+'/vertex_radius_array.npy')

    # FILE.save_npy(trees_folder+"/mask.npy", mask)
    # FILE.save_npy(trees_folder+"/changes.npy", changes)


def preprocess(V, E, R):
    V, E, R = SPLIT.ensure_cardinality(V, E, R)
    save(V, E, R, None, None, "split")
    # V, E, R, mask = ARTE.generate_arterial(V, E, R)
    # save(V, E, R, mask, None, "arterial")
    regSave = lambda V, E, R, chg, i: save(V, E, R, [], chg, f"-reg{i}")
    V, E, R, changes = SMOOTH.smooth(V, E, R, iter=201, lap_lr=0.1, rep_lr=0.05, save_fn=regSave)
    return V, E, R, [], changes

if __name__ == '__main__':
    tree_folder = f"../../data/trees/__reg500"
    V, E, R = DATA.load_skeleton_data(tree_folder)

    V, E, R, mask, changes = preprocess(V, E, R)

    save(V, E, R, mask, changes, "-regDone")
