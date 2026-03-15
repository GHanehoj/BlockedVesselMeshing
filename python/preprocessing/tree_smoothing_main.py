import sys
import os
sys.path.append(os.path.abspath('../'))
import preprocessing.tree_smoothing as SMOOTH
import tools.file as FILE

data_path = "../../data/"

def save(V, E, R, changes, folder):
    FILE.create_folders_if_not_exist(folder)
    FILE.save_skeleton_data(V, E, R, folder)
    FILE.save_npy(folder+"/changes.npy", changes)


### Brain ###
skeleton_folder = data_path+"/input/brain/skeleton/"
preprocess_folder = data_path+"/input/brain/preprocessed/"
V, E, R = FILE.load_skeleton_data(skeleton_folder)

regSave = lambda V, E, R, chg, i: save(V, E, R, [], chg, preprocess_folder+f"/reg{i}/")
V, E, R, changes = SMOOTH.smooth(V, E, R, iter=101, lap_lr=0.04, rep_lr=0.02, bar_lr=25, save_fn=regSave)

save(V, E, R, changes, preprocess_folder+"/regDone/")


### Kidney ###
skeleton_folder = data_path+"/input/kidney/skeleton/"
preprocess_folder = data_path+"/input/kidney/preprocessed/"
V, E, R = FILE.load_skeleton_data(skeleton_folder)

regSave = lambda V, E, R, chg, i: save(V, E, R, [], chg, preprocess_folder+f"/reg{i}/")
V, E, R, changes = SMOOTH.smooth(V, E, R, iter=101, lap_lr=0.04, rep_lr=0.02, bar_lr=25, save_fn=regSave)

save(V, E, R, changes, preprocess_folder+"/regDone/")


### Liver ###
skeleton_folder = data_path+"/input/liver/skeleton/"
preprocess_folder = data_path+"/input/liver/preprocessed/"
V, E, R = FILE.load_skeleton_data(skeleton_folder)

regSave = lambda V, E, R, chg, i: save(V, E, R, [], chg, preprocess_folder+f"/reg{i}/")
V, E, R, changes = SMOOTH.smooth(V, E, R, iter=101, lap_lr=0.04, rep_lr=0.02, bar_lr=25, save_fn=regSave)

save(V, E, R, changes, preprocess_folder+"/regDone/")


### Lung ###
skeleton_folder = data_path+"/input/lung/skeleton/"
preprocess_folder = data_path+"/input/lung/preprocessed/"
V, E, R = FILE.load_skeleton_data(skeleton_folder)

regSave = lambda V, E, R, chg, i: save(V, E, R, [], chg, preprocess_folder+f"/reg{i}/")
V, E, R, changes = SMOOTH.smooth(V, E, R, iter=101, lap_lr=0.04, rep_lr=0.02, bar_lr=25, save_fn=regSave)

save(V, E, R, changes, preprocess_folder+"/regDone/")


### Tree ###
skeleton_folder = data_path+"/input/tree/skeleton/"
preprocess_folder = data_path+"/input/tree/preprocessed/"
V, E, R = FILE.load_skeleton_data(skeleton_folder)

regSave = lambda V, E, R, chg, i: save(V, E, R, [], chg, preprocess_folder+f"/reg{i}/")
V, E, R, changes = SMOOTH.smooth(V, E, R, iter=101, lap_lr=0.04, rep_lr=0.02, bar_lr=25, save_fn=regSave)

save(V, E, R, changes, preprocess_folder+"/regDone/")

