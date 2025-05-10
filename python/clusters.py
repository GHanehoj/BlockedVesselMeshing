import numpy as np
from tools.numpy_util import angle_between, lerp, normalize
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
        return 1
    else:
        return 1 + np.sum([_cnt(outflow.cluster, depth+1, max_depth) for outflow in cluster.outflows], dtype=int)

class Cluster:
    def __init__(self, node, cut_dist):
        self.nodes = [node]
        self.leaves = []
        self.outflows = []
        if cut_dist is None:
            assert(node.parent.parent is None)
            cut_dist = np.linalg.norm(node.position-node.parent.position)
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

    def adjust_inflow(self, adjustment):
        node = self.nodes[0]
        parent = node.parent

        orig_cut_dist = self.in_data.cut_dist
        new_cut_dist = orig_cut_dist + adjustment
        self.in_data = FlowData(node, parent, new_cut_dist)

        connector = parent.position - node.position
        buffer = node.radius
        frac = (new_cut_dist+buffer)/np.linalg.norm(connector)
        pos = lerp(node.position, parent.position, np.minimum(1, frac))
        rad = lerp(node.radius, parent.radius, np.minimum(1,frac))

        self.V[1] = pos
        self.R[1] = rad

def make_cluster(node, cut_dist=None):
    cluster = Cluster(node, cut_dist)
    expand_cluster(cluster, node, 0)
    return cluster


def expand_cluster(cluster, node, node_idx):
    for child in node.children:
        cut_dist, slack, child_cut_dist = calc_cut_dist(node, child)
        if len(child.children) == 0:
            cluster.add_leaf(node, child, node_idx, cut_dist)
            continue
        if cut_dist is not None and not has_self_intersection(cluster, node, child, cut_dist):
            child_cluster = make_cluster(child, child_cut_dist)
            inflow_adjustment = calc_inflow_adjustment(child_cluster, slack)
            if inflow_adjustment is not None:
                if inflow_adjustment != 0:
                    child_cluster.adjust_inflow(inflow_adjustment*slack)
                cluster.add_outflow(child_cluster, node, child, node_idx, cut_dist)
                continue
        child_idx = cluster.add_node(child, node_idx)
        expand_cluster(cluster, child, child_idx)


def calc_cut_dist(node, child):
    connector = child.position - node.position
    dist = np.linalg.norm(connector)

    min_out_len = min_len_all(node, child)
    min_in_len = min_len_all(child, node)

    if np.any(dist < 1.1*(min_out_len + min_in_len)): # just a number??
        return None, None, None

    slack = np.min(dist - 1.1*(min_out_len + min_in_len))

    return min_out_len, slack, min_in_len

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

def has_self_intersection(cluster : Cluster, node, child, cut_dist):
    flow_data = FlowData(node, child, cut_dist)
    for i in range(len(cluster.E)):
        e = cluster.E[i]
        p1, p2 = cluster.V[e]
        if np.allclose(p1, node.position) or np.allclose(p2, node.position):
            continue
        r1, r2 = cluster.R[e]
        dist, t = point_seg_intersect(flow_data.point, p1, p2)
        r = lerp(r1, r2, t)
        if dist < 1.5*(r + flow_data.radius):
            return True
    return False
def calc_inflow_adjustment(cluster, slack):
    flow_data = cluster.in_data
    for i in np.linspace(0, 1, 5):
        point = flow_data.point + i*slack*flow_data.dir
        if not has_inflow_intersection(cluster, point, flow_data.radius): # Assume radius difference is neglegible for now. TODO
            return i
    return None

def has_inflow_intersection(cluster, inflow_point, inflow_radius):
    for i in range(len(cluster.E)):
        e = cluster.E[i]
        p1, p2 = cluster.V[e]
        if np.allclose(p1, cluster.nodes[0].position) or np.allclose(p2, cluster.nodes[0].position):
            continue
        r1, r2 = cluster.R[e]
        dist, t = point_seg_intersect(inflow_point, p1, p2)
        r = lerp(r1, r2, t)
        if dist < 1.5*(r + inflow_radius):
            return True
    return False

def point_seg_intersect(pnt, start, end):
    line = end-start
    vec = pnt-start
    l = np.linalg.norm(line)
    dir = line/l
    t = np.dot(dir, vec)/l
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = start + t*l*dir
    dist = np.linalg.norm(nearest-pnt)
    return (dist, t)


def cluster_list(root : Cluster):
    child_clusters = [cluster for outflow in root.outflows for cluster in cluster_list(outflow.cluster)]
    return [root] + child_clusters
