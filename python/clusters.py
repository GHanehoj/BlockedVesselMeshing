import numpy as np
from tools.numpy_util import angle_between, lerp
from collections import namedtuple
OutFlow = namedtuple("OutFlow", ["cluster", "data"])

class FlowData:
    def __init__(self, node, other, cut_dist):
        diff = other.position - node.position
        dist = np.linalg.norm(diff)
        frac = cut_dist/dist

        self.cut_dist = cut_dist
        self.dir = diff/dist
        self.point = lerp(node.position, other.position, frac)
        self.radius = lerp(node.radius, other.radius, frac)

def _search(cluster, s, idx):
    if cluster.nodes[0].index == idx:
        print(s)
    else:
        for i, outflow in enumerate(cluster.outflows):
            _search(outflow.cluster, s+f".outflows[{i}].cluster", idx)

def _cnt(cluster, depth, max_depth):
    if depth > max_depth:
        return 0
    else:
        return 1 + np.sum([_cnt(outflow.cluster, depth+1, max_depth) for outflow in cluster.outflows], dtype=int)

class Cluster:
    def __init__(self, node):
        self.nodes = [node]
        self.leaves = []
        self.outflows = []
        cut_dist = calc_cut_dist(node, node.parent)
        if cut_dist is None:
            if node.parent.parent is None:
                cut_dist = np.linalg.norm(node.position-node.parent.position)
            else:
                assert(False)
        self.in_data = FlowData(node, node.parent, cut_dist)
        self.V = np.array([node.position])
        self.E = np.empty((0,2), dtype=int)
        self.R = np.array([node.radius])
        self._add_flow_to_graph(node, node.parent, 0, cut_dist)

    def add_node(self, node, parent_idx):
        self.nodes += [node]
        node_idx = len(self.V)
        self.V = np.concatenate((self.V, [node.position]), axis=0)
        self.E = np.concatenate((self.E, [[parent_idx, node_idx]]), axis=0)
        self.R = np.concatenate((self.R, [node.radius]), axis=0)
        return node_idx

    def add_leaf(self, node, leaf, node_idx, cut_dist):
        self.leaves += [leaf]
        if cut_dist is None:
            cut_dist = np.linalg.norm(leaf.position - node.position)
        self._add_flow_to_graph(node, leaf, node_idx, cut_dist)

    def add_outflow(self, other_cluster, node, other, node_idx, cut_dist):
        self.outflows += [OutFlow(other_cluster, FlowData(node, other, cut_dist))]
        self._add_flow_to_graph(node, other, node_idx, cut_dist)

    def _add_flow_to_graph(self, node, other, node_idx, cut_dist):
        other_idx = len(self.V)
        connector = other.position - node.position
        buffer = node.radius
        frac = (cut_dist+buffer)/np.linalg.norm(connector)
        pos = lerp(node.position, other.position, np.minimum(1, frac))
        rad = lerp(node.radius, other.radius, np.minimum(1,frac))
        self.V = np.concatenate((self.V, [pos]), axis=0)
        self.E = np.concatenate((self.E, [[node_idx, other_idx]]), axis=0)
        self.R = np.concatenate((self.R, [rad]), axis=0)

    def calc_dx(self, res):
        min_rad = np.min(self.R)
        return (2*min_rad)/res


def make_cluster(node):
    cluster = Cluster(node)
    expand_cluster(cluster, node, 0)
    return cluster


def expand_cluster(cluster, node, node_idx):
    for child in node.children:
        cut_dist = calc_cut_dist(node, child)
        if len(child.children) == 0:
            cluster.add_leaf(node, child, node_idx, cut_dist)
            continue
        if cut_dist is not None:
            child_cluster = make_cluster(child)
            cluster.add_outflow(child_cluster, node, child, node_idx, cut_dist)
        else:
            child_idx = cluster.add_node(child, node_idx)
            expand_cluster(cluster, child, child_idx)


def calc_cut_dist(node, child):
    connector = child.position - node.position
    dist = np.linalg.norm(connector)

    min_out_len = min_len_all(node, child)
    min_in_len = min_len_all(child, node)

    if np.any(dist < 1.1*(min_out_len + min_in_len)):
        return None

    return min_out_len


def min_len_all(node, child):
    all_neighbours = node.children + ([node.parent] if node.parent is not None else [])
    return np.max([2*node.radius]+[min_len_ang(node, child, other) for other in all_neighbours if child != other])

def min_len_ang(node, child, other):
    child_conn = child.position - node.position
    other_conn = other.position - node.position

    theta = angle_between(child_conn, other_conn)

    if theta > np.pi/2: return 0
    if abs((child.radius-node.radius)/np.linalg.norm(child_conn)) > 1: return 0
    if abs((other.radius-node.radius)/np.linalg.norm(other_conn)) > 1: return 0

    child_ang_offset = np.arcsin((child.radius-node.radius)/np.linalg.norm(child_conn))
    other_ang_offset = np.arcsin((other.radius-node.radius)/np.linalg.norm(other_conn))

    phi = (np.pi+child_ang_offset+other_ang_offset-theta)/2

    l = node.radius/np.cos(phi)

    a = np.pi/2+child_ang_offset-phi

    split_dist = np.cos(a)*l

    return split_dist + 1*node.radius

def cluster_list(root : Cluster):
    child_clusters = [cluster for outflow in root.outflows for cluster in cluster_list(outflow.cluster)]
    return [root] + child_clusters
