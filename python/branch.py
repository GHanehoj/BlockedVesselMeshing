import numpy as np
import math
import data as DATA
import tree as TREE
import rainbow.math.quaternion as QUAT
import rainbow.math.vector3 as VEC
def branch_name(idx): return f"b{idx}"

### Normalized Branch ###
class FullBranch:
    def __init__(self, node, tree_conf):
        if (len(node.children) != 2): raise "only supports 3-arms"
        biggest_child_idx = int(node.children[0].radius < node.children[1].radius)

        self.idx = node.index
        self.tree_conf = tree_conf

        # order: p, c, i, j
        self.arms      = [0, 2, 3]
        self.indices   = np.array([node.parent.index,
                                   node.index,
                                   node.children[biggest_child_idx].index,
                                   node.children[1-biggest_child_idx].index])
        self.positions = np.array([node.parent.position,
                                   node.position,
                                   node.children[biggest_child_idx].position,
                                   node.children[1-biggest_child_idx].position])
        self.radii     = np.array([node.parent.radius,
                                   node.radius,
                                   node.children[biggest_child_idx].radius,
                                   node.children[1-biggest_child_idx].radius])

        self.T = -self.positions[1]
        R1 = QUAT.R_vector_to_vector(self.positions[0]+self.T, -VEC.k())
        R2 = QUAT.R_vector_into_xz_plane(QUAT.rotate(R1, self.positions[2]+self.T))
        self.R = QUAT.prod(R2,R1)
        self.S = 1 / np.linalg.norm(self.positions[1] - self.positions[0])

        self.norm_positions = self.normalize(self.positions)
        self.norm_radii = self.radii * self.S


    def normalize(self, v):
        return QUAT.rotate(self.R, v + self.T) * self.S

    def denormalize(self, v):
        return QUAT.rotate(QUAT.conjugate(self.R), v / self.S) - self.T

    def get_hyperparams(self):
        v_i = self.norm_positions[2]
        v_j = self.norm_positions[3]

        l_i = np.linalg.norm(v_i)
        l_j = np.linalg.norm(v_j)

        alpha = math.atan2(v_i[2], v_i[0])
        beta = math.atan2(v_j[1], v_j[0])
        gamma = math.atan2(np.linalg.norm(v_j[0:2]), v_j[2])

        return np.concatenate((self.norm_radii, [l_i, l_j], [alpha, beta, gamma]))

    def create_local_graph(self):
        V = self.norm_positions
        E = np.array([[0, 1], [1, 2], [1, 3]])
        R = self.norm_radii
        return V, E, R


### Normalized Branch Closeup ###
close_range_radii = 3
class CloseBranch:
    def __init__(self, node, tree_conf):
        if (len(node.children) != 2): raise "only supports 3-arms"
        biggest_child_idx = int(node.children[0].radius < node.children[1].radius)

        self.idx = node.index
        self.tree_conf = tree_conf

        # order: p, c, i, j
        self.arms      = [0, 2, 3]
        self.indices   = np.array([node.parent.index,
                                   node.index,
                                   node.children[biggest_child_idx].index,
                                   node.children[1-biggest_child_idx].index])
        self.positions = np.array([node.parent.position,
                                   node.position,
                                   node.children[biggest_child_idx].position,
                                   node.children[1-biggest_child_idx].position])
        self.radii     = np.array([node.parent.radius,
                                   node.radius,
                                   node.children[biggest_child_idx].radius,
                                   node.children[1-biggest_child_idx].radius])

        self.T = -self.positions[1]
        R1 = QUAT.R_vector_to_vector(self.positions[0]+self.T, -VEC.k())
        R2 = QUAT.R_vector_into_xz_plane(QUAT.rotate(R1, self.positions[2]+self.T))
        self.R = QUAT.prod(R2,R1)
        self.S = 1 / self.radii[1]

        self.norm_positions = self.normalize(self.positions)
        self.norm_radii = self.radii * self.S

        norm_arm_lengths = np.linalg.norm(self.norm_positions[self.arms], axis=1)
        arm_len_fracs = 1/norm_arm_lengths * close_range_radii

        self.norm_positions[self.arms] *= arm_len_fracs[..., np.newaxis]
        self.norm_radii[self.arms] = arm_len_fracs*self.norm_radii[self.arms] + (1-arm_len_fracs)
        if (np.any(self.norm_radii[self.arms] < 0)):
            print(f"WARNING: arm in branch {self.idx} got extended s.t. end radius went below 0.")


    def normalize(self, v):
        return QUAT.rotate(self.R, v + self.T) * self.S

    def denormalize(self, v):
        return QUAT.rotate(QUAT.conjugate(self.R), v / self.S) - self.T

    def get_hyperparams(self):
        v_i = self.norm_positions[2]
        v_j = self.norm_positions[3]

        alpha = math.atan2(v_i[2], v_i[0])
        beta = math.atan2(v_j[1], v_j[0])
        gamma = math.atan2(np.linalg.norm(v_j[0:2]), v_j[2])

        return np.concatenate((self.norm_radii[self.arms], [alpha, beta, gamma]))

    def create_local_graph(self):
        V = self.norm_positions
        E = np.array([[0, 1], [1, 2], [1, 3]])
        R = self.norm_radii
        return V, E, R


### Construction from tree ###
def mk_branch(node, tree_conf):
    if tree_conf.close:
        return CloseBranch(node, tree_conf)
    else:
        return FullBranch(node, tree_conf)

def get_branch_by_idx(idx, tree_conf):
    tree_folder = f"../../data/trees/{tree_conf.id}"
    V, E, R = DATA.load_skeleton_data_def(tree_folder)
    _, nodes = TREE.make_tree(V, E, R)
    assert(len(nodes[idx].children) == 2)
    return mk_branch(nodes[idx], tree_conf)

def get_all_branches(tree_conf):
    tree_folder = f"../../data/trees/{tree_conf.id}"
    V, E, R = DATA.load_skeleton_data_def(tree_folder)
    _, nodes = TREE.make_tree(V, E, R)
    nodes_3_branch = [node for node in nodes if len(node.children) == 2]
    return [mk_branch(node, tree_conf) for idx, node in enumerate(nodes_3_branch)]