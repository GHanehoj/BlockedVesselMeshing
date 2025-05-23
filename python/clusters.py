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
        self.outflows = []
        self.V = np.array([node.position])
        self.E = np.empty((0,2), dtype=int)
        self.R = np.array([node.radius])
        if cut_dist is None:
            self.in_data = None
        else:
            self.in_data = FlowData(node, node.parent, cut_dist)
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

def make_cluster(node, done_f=None, cut_dist=None):
    cluster = Cluster(node, cut_dist)
    if node.index == 5058:
        a=2
    expand_cluster(cluster, node, 0, done_f)
    return cluster



    # def _expand_cluster(node, node_idx):
    #     for child in node.children:
    #         cut_dist, slack, child_cut_dist = calc_cut_dist(node, child)
    #         if cut_dist is not None and not has_self_intersection(cluster, node, child, cut_dist):
    #             child_cluster = make_cluster(child, done_f, child_cut_dist)
    #             inflow_adjustment = calc_inflow_adjustment(child_cluster, slack)
    #             if inflow_adjustment is not None:
    #                 if inflow_adjustment != 0:
    #                     child_cluster.adjust_inflow(inflow_adjustment*slack)
    #                 cluster.add_outflow(child_cluster, node, child, node_idx, cut_dist)

    #                 continue
    #         child_idx = cluster.add_node(child, node_idx)
    #         expand_cluster(cluster, child, child_idx, done_f)

def expand_cluster(cluster, node, node_idx, done_f):
    outflows = []
    def _expand_cluster(node, node_idx):
        for child in node.children:
            cut_dist, slack, child_cut_dist = calc_cut_dist(node, child)
            if cut_dist is not None:
                outflows.append((node, node_idx, child, cut_dist, slack, child_cut_dist, FlowData(node, child, cut_dist), FlowData(child, node, child_cut_dist)))
            else:
                child_idx = cluster.add_node(child, node_idx)
                _expand_cluster(child, child_idx)

    _expand_cluster(node, node_idx)
    
    outflow_adjustments = []
    child_clusters = []
    child_inflow_adjustments = []

    def update_outflow_adjustments():
        outflow_adjustments.clear()
        for idx in range(len(outflows)):
            outflow_adjustments.append(calc_outflow_adjustment(cluster, outflows, idx))
    def update_children():
        for idx in range(len(child_clusters), len(outflows)):
            (_, _, child, _, _, child_cut_dist, _, _) = outflows[idx]
            child_clusters.append(make_cluster(child, done_f, child_cut_dist))
        for idx in range(len(outflow_adjustments), len(outflows)):
            outflow_adjustments.append(calc_outflow_adjustment(cluster, outflows, idx))

    def update_inflow_adjustments():
        child_inflow_adjustments.clear()
        for child_cluster, (_, _, _, _, slack, _, _, _), out_adj in zip(child_clusters, outflows, outflow_adjustments, strict=True):
            child_inflow_adjustments.append(calc_inflow_adjustment(child_cluster, slack*(1-out_adj)))


    def adjust():
        update_outflow_adjustments()
        while np.any(np.isnan(outflow_adjustments)):
            idx = np.argmax(np.isnan(outflow_adjustments))
            (_, node_idx, child, _, _, _, _, _) = outflows[idx]
            del outflows[idx]
            if len(child_clusters) > idx: del child_clusters[idx]

            child_idx = cluster.add_node(child, node_idx)
            _expand_cluster(child, child_idx)

            update_outflow_adjustments()

        update_children()

        did_change = False
        update_inflow_adjustments()
        while np.any(np.isnan(child_inflow_adjustments)):
            did_change = True
            idx = np.argmax(np.isnan(child_inflow_adjustments))
            (_, node_idx, child, _, _, _, _, _) = outflows[idx]
            del child_clusters[idx]
            del outflows[idx]
            del outflow_adjustments[idx]

            child_idx = cluster.add_node(child, node_idx)
            _expand_cluster(child, child_idx)

            update_children()

            update_inflow_adjustments()
        if did_change: adjust()
    adjust()

    if done_f is not None:
        for _ in cluster.nodes:
            done_f()

    for child_cluster, (node, node_idx, child, cut_dist, slack, _, _, _), out_adj, in_adj in zip(child_clusters, outflows, outflow_adjustments, child_inflow_adjustments, strict=True):
        if in_adj != 0:
            child_cluster.adjust_inflow(in_adj*slack*(1-out_adj))
        cluster.add_outflow(child_cluster, node, child, node_idx, cut_dist + out_adj*slack)
    ## TODO: outflows can still collide with each other.

def calc_cut_dist(node, child):
    connector = child.position - node.position
    dist = np.linalg.norm(connector)

    min_out_len = min_len_all(node, child)
    min_in_len = min_len_all(child, node)
    # min_gap = 0.2*(node.radius+child.radius)/2
    min_gap = 0.3*(node.radius+child.radius)/2
    slack = dist - (min_out_len + min_in_len + min_gap)

    if slack < 0: return None, None, None

    return min_out_len, slack, min_in_len

def min_len_all(node, child):
    all_neighbours = node.children + ([node.parent] if node.parent is not None else [])
    return np.max([1.5*node.radius]+[min_len_ang(node, child, other) for other in all_neighbours if child != other])

def min_len_ang(node, child, other):
    child_conn = child.position - node.position
    other_conn = other.position - node.position

    theta = 0.7*angle_between(child_conn, other_conn)

    if theta > np.pi/2: return 0
    if abs((node.radius-child.radius)/np.linalg.norm(child_conn)) > 1: return 0
    if abs((node.radius-other.radius)/np.linalg.norm(other_conn)) > 1: return 0

    child_ang_offset = np.arcsin(abs((node.radius-child.radius))/np.linalg.norm(child_conn))
    other_ang_offset = np.arcsin(abs((node.radius-other.radius))/np.linalg.norm(other_conn))

    phi = (np.pi-child_ang_offset-other_ang_offset-theta)/2

    d = node.radius/np.cos(phi)

    psi = (np.pi+child_ang_offset-other_ang_offset-theta)/2
    psi2 = phi+child_ang_offset

    if not np.isclose(psi, psi2):
        a=2

    split_dist = np.sin(psi)*d

    return 1.2*split_dist

def calc_outflow_adjustment(cluster, outflows, idx):
    (node, _, _, _, slack, _, flow_data, _) = outflows[idx]
    other_flows = [(outflow[0], outflow[6]) for i, outflow in enumerate(outflows) if i != idx]
    for i in np.linspace(0, 1, 5):
        point = flow_data.point + i*slack*flow_data.dir
        if not has_flow_intersection(cluster, other_flows, node, point, flow_data.radius): # Assume radius difference is neglegible for now. TODO
            return i
    return np.nan

def calc_inflow_adjustment(cluster, slack):
    flow_data = cluster.in_data
    for i in np.linspace(0, 1, 5):
        point = flow_data.point + i*slack*flow_data.dir
        if not has_flow_intersection(cluster, [], cluster.nodes[0], point, flow_data.radius): # Assume radius difference is neglegible for now. TODO
            return i
    return np.nan

def has_flow_intersection(cluster : Cluster, other_flows, node, flow_point, flow_radius):
    for i in range(len(cluster.E)):
        e = cluster.E[i]
        p1, p2 = cluster.V[e]
        if np.allclose(p1, node.position) or np.allclose(p2, node.position):
            continue
        r1, r2 = cluster.R[e]
        dist, t = point_seg_intersect(flow_point, p1, p2)
        r = lerp(r1, r2, t)
        if dist < 1.5*(r + flow_radius):
            return True
    for i in range(len(other_flows)):
        other_node, other_flow_data = other_flows[i]
        if node == other_node: continue
        p1 = other_node.position
        p2 = other_flow_data.point
        r1 = other_node.radius
        r2 = other_flow_data.radius
        dist, t = point_seg_intersect(flow_point, p1, p2)
        if dist < (r2 + flow_radius):
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


def cluster_stats(root : Cluster, res):
    clusters = cluster_list(root)
    stats = np.empty((len(clusters), 2))
    for i, cluster in enumerate(clusters):
        stats[i,0] = size_estimate(cluster, res)
        stats[i,1] = len(cluster.nodes)
    return stats

def size_estimate(cluster, res):
    tot = 0
    radii = ([cluster.in_data.radius] if cluster.in_data is not None else []) + [node.radius for node in cluster.nodes] + [outflow.data.radius for outflow in cluster.outflows]
    dx = ((2*np.min(radii))/res)
    for e in cluster.E:
        r = np.mean(cluster.R[e])
        l = np.linalg.norm(cluster.V[e[0]]-cluster.V[e[1]])
        V = l*np.pi*r*r
        n = V/(dx*dx*dx)
        tot += n
    return tot