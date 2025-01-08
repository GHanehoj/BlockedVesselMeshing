import sys
import os
sys.path.append(os.path.abspath('../'))
import preprocessing.vertex_splitter as SPLIT
import preprocessing.tree_smoothing as SMOOTH
import preprocessing.arterial_generation as ARTE
import data as DATA
import tools.file as FILE

def load():
    tree_folder = f"../../data"
    V, E, R = DATA.load_skeleton_data_def(tree_folder)
    return V, E, R

def save(V, E, R, changes, id):
    trees_folder = os.path.dirname(__file__)+"/../../data/trees/"+id
    FILE.save_skeleton_data(V, E, R,
                            vertex_array_file  = trees_folder+'/vertex_array.npy',
                            edge_array_file    = trees_folder+'/edge_array.npy',
                            vertex_radius_file = trees_folder+'/vertex_radius_array.npy')

    FILE.save_npy(trees_folder+"/changes.npy", changes)


def preprocess(V, E, R):
    V, E, R = SPLIT.ensure_cardinality(V, E, R)
    V, E, R, mask = ARTE.generate_arterial(V, E, R)
    regSave = lambda V, E, R, chg, i: save(V, E, R, chg, f"reg{i}")
    return SMOOTH.smooth(V, E, R, save_fn=regSave)

if __name__ == '__main__':
    V, E, R = load()

    V, E, R, changes = preprocess(V, E, R)

    save(V, E, R, changes, "regDone")
