import numpy as np
from tools.numpy_util import angle_between, lerp
from collections import namedtuple
OutFlow = namedtuple("OutFlow", ["cluster", "data"])

_rad_mul = 2.2
_buffer = 2

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
        self.outflows = []
        cut_dist = calc_cut_dist(node, -1, node.parent)
        if cut_dist is None:
            cut_dist = np.linalg.norm(node.position-node.parent.position)-1.1*_rad_mul*node.parent.radius
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

    def add_outflow(self, other_cluster, node, other, node_idx, cut_dist):
        self.outflows += [OutFlow(other_cluster, FlowData(node, other, cut_dist))]
        self._add_flow_to_graph(node, other, node_idx, cut_dist)

    def _add_flow_to_graph(self, node, other, node_idx, cut_dist):
        other_idx = len(self.V)
        connector = other.position - node.position
        frac = _buffer*(cut_dist)/np.linalg.norm(connector)
        pos = lerp(node.position, other.position, np.minimum(1, frac))
        rad = lerp(node.radius, other.radius, np.minimum(1,frac))
        self.V = np.concatenate((self.V, [pos]), axis=0)
        self.E = np.concatenate((self.E, [[node_idx, other_idx]]), axis=0)
        self.R = np.concatenate((self.R, [rad]), axis=0)

    def calc_dx(self, res):
        node_rad = [node.radius for node in self.nodes]
        out_rad = [outflow.data.radius for outflow in self.outflows]
        in_rad = self.in_data.radius

        min_rad = np.min(np.concatenate((node_rad, out_rad, [in_rad])))

        return (2*min_rad)/res


def make_cluster(node):
    cluster = Cluster(node)
    expand_cluster(cluster, node, 0)
    return cluster


def expand_cluster(cluster, node, node_idx):
    for i, child in enumerate(node.children):
        cut_dist = calc_cut_dist(node, i, child)
        if cut_dist is not None:
            child_cluster = make_cluster(child)
            cluster.add_outflow(child_cluster, node, child, node_idx, cut_dist)
        else:
            child_idx = cluster.add_node(child, node_idx)
            expand_cluster(cluster, child, child_idx)


def calc_cut_dist(node, i, child):
    connector = child.position - node.position
    dist = np.linalg.norm(connector)

    if dist < 1.1*_rad_mul*(node.radius+child.radius):
        return None

    if len(node.children) == 0:
        return _rad_mul*node.radius

    other_connectors = np.empty((len(node.children), 3))
    other_connector_radii = np.empty((len(node.children)))

    for j, other_node in enumerate(node.children):
        if j == i: other_node = node.parent
        other_connectors[j] = other_node.position - node.position
        other_connector_radii[j] = other_node.radius

    angles = angle_between(other_connectors, connector)
    min_lens = (np.where(angles < np.pi/2,
                         1.5*((node.radius+other_connector_radii)/(2*np.sin(angles))+(node.radius+child.radius)/(2*np.tan(angles))),
                         0.0))
    if np.any(dist < 1.1*(min_lens + _rad_mul*child.radius)):
        return None

    return np.maximum(_rad_mul*node.radius, np.max(min_lens))