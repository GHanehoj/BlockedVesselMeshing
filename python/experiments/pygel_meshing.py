import sys
import os
sys.path.append(os.path.abspath('../'))
import data as DATA
import tree as TREE
from tools.mesh_util import TriMesh
import pygel3d.graph as pgG
import pygel3d.hmesh as pgM
import numpy as np
import meshio


def filter_by_idxs(V,E,R, idxs):
    vertex_mask = np.in1d(np.arange(len(V)), idxs)
    edge_mask = np.all(np.isin(E, idxs), axis=1)
    offsets = np.cumsum(~vertex_mask)
    E_new = E[edge_mask]
    E_new -= offsets[E_new]
    return V[vertex_mask], E_new, R[vertex_mask]

def subdivide(V,E,R, n):
    if n == 0: return V.copy(), E.copy(), R.copy()
    V2 = V.copy()
    E2 = []
    R2 = R.copy()
    for e in E:
        v0 = V[e[0]]
        v1 = V[e[1]]
        r0 = R[e[0]]
        r1 = R[e[1]]
        ts = np.linspace(0, 1, n+1, False)[1:]
        idx0 = len(V2)
        V2 = np.concatenate((V2, v0 + ts[:, None]*(v1-v0)), axis=0)
        R2 = np.concatenate((R2, r0 + ts*(r1-r0)))

        E2.append([e[0], idx0])
        for i in range(n-1):
            E2.append([idx0+i, idx0+i+1])
        E2.append([idx0+n-1, e[1]])

    return V2, np.array(E2), R2


def feq(V, E, R, sd):
    V, E, R = subdivide(V, E, R, sd)
    g = pgG.Graph()
    print("subdivided")
    for v in V:
        g.add_node(v)
    print("nodes added")
    for e in E:
        g.connect_nodes(e[0], e[1])
    print("edges added")
    mesh = pgM.skeleton_to_feq(g, R)
    pgM.obj_save(f"feq_sd{sd}_cc0.obj", mesh)
    print("feq done")
    for i in range(2):
        pgM.cc_subdivide(mesh)
        pgM.obj_save(f"feq_sd{sd}_cc{i+1}.obj", mesh)
    print("cc done")


idxs = [27781,28125,28139,34885,27975,51806,49294,36658,28321,53666,36294,28490,28641,53665,37338,40926,39657,27786,27792,33600,38481,31967,27401,50632,27592,27597,34503,27408,27797,30650,29936,29959,30656,27812,31270,27996,47546,48490,31971,28328,50635,48489,27992,34183,48463,31981,27811,50645,32442,50644,48462,28327,43565,28154,45397,52241,30681,29635,29931,26990,48457,29741,48456,48119,30958,30286,49621,32230,28009,32238,49324,29769,29797,27626,48305,31079,33226,49752,49323,30856,27609,49524,31249,27214,49923,34196,27603,48458,32012,29692,29640,29918,29721,48446,48127,32228,34374,48186,29744,29933,48247,48185,30266,31022,50669,29775,29644,29706,48132,48277,30835,32128,32416,30420,25697,49038,31058,33245,49729,29932,29826,37067,29994,31924,27802,50589,48527,34150,37008,53952,30370,31546,48018,32082,32361,50741,48335,39848,30074,34075,48624,55372,36588,39884,42023,27782,56452,42048,43904,48426,53994,46284,30657,31269,27604,49946,46651,36656,58795,58589,37112,30959,53995,46754,42245,44005,46419,32772,38972,35374,54807,51255,44463,46907,38405,40445,28123,55632,54522,44830,54521,29899,46326,30145,30428,29768,28308,48271,49044,33771,36782,34011,32641,29269,51163,52114,34612,43338,58613,30855,34178,27790,52228,49523,36550,39599,34326,30858,41922,49527,52366,34313,52365,37111,29626,29754,29728,29724,29636,48220,48226,29790,32161,48255,29761,29698,48261,37089,30180,53978,48110,29945,29711,29680,48203,48476,29952,45190,30271,29702,29648,48194,48871,29905,48870,29633,2,29622,29756,45609,35129,58230,45916,37859,58368,30274,29710,29809,48202,48873,30254,39754,29624,29755,29727,30172,48225,48256,30136,30847,29946,30668,29682,49312,48478,31979,48477,4,29650,29881,32024,23616,50682,23617,23990,48406,34227,30727,30747,45192,30851,58008,34369,29856,46788,45398,58884,46878,33955,58932,52401,30919,29873,49584,34389,38399,45552,8,10,3,7,6,29621,29631,29629,29751,48113,29625,30272,29951,48872,32234,29760,50882,29623,29947,29701,30898,48193,48479,29904,30667,34378,29678,29836,48168,52408,30135,34379,11,12,5,29627,29753,29729,48254,48111,30270,30963,29628,29752,45610,45917,48112,29944,30962,29620,29950,29725,29792,48221,48480,29920,30666,48108,30275,29709,48874,30846,29700,49517,9,1]


V, E, R = DATA.load_skeleton_data("../../data/trees/_reg200")
root, _ = TREE.make_tree(V, E, R)

def _search(node, s, idx):
    if node.index == idx:
        print(s)
    else:
        for i, child in enumerate(node.children):
            _search(child, s+f".children[{i}]", idx)

def _cnt(node):
    return 1 + np.sum([_cnt(child) for child in node.children], dtype=int)

def _idxs(node):
    if len(node.children) == 0: return [node.index]
    return np.concatenate(([node.index], *[_idxs(child) for child in node.children]))

node = root.children[0].children[0].children[1].children[0].children[0].children[1]

idxs = _idxs(node)

V,E,R = filter_by_idxs(V,E,R,idxs)
a=2

feq(V,E,R, 1)

# feq(V,E,R, 2)

# feq(V,E,R, 4)

# from tools.pyvista_plotting import *
# show_tri_mesh(feq_tri)


# subdivisions = [0, 1, 2, 4, 8]
# CCs = [0, 1, 2]


