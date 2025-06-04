import numpy as np
from tools.rdp import rdp

class TreeNode:
    """
    This class represents a skeleton node of a renal aerial vascular
    vessel structure. A node knows about its parent (up-stream direction
    of blood flow) and its children (blood flow down-stream).
    """

    def __init__(self):
        self.index = -1         # The vertex index of this node.
        self.parent = None      # Reference to parent node of this node.
        self.children = []      # References to all child nodes of this node.

        self.position = None    # The position of this node.
        self.radius = None      # The radius of this node.

def make_tree(V, E, R):
    """
    Factory function for creating a tree data structure of an edge-indexed array representation of a vascular
    skeleton tree instance.

    :param V:   N-by-3 vertex coordinate array. First vertex is assumed to be the root of the tree.
    :param E:   K-by-2 edge vertex index array. Edges are assumed direction going from parent to child nodes.
    :return:    A reference to the root node as well as a list of all nodes in the tree.
    """
    nodes = [TreeNode() for _ in V]
    for i in range(len(V)):
        nodes[i].index = i
        nodes[i].position = V[i]
        nodes[i].radius = R[i]
    for e in E:
        parent_idx = e[0]
        child_idx = e[1]
        parent = nodes[parent_idx]
        child = nodes[child_idx]
        child.parent = parent
        parent.children.append(child)
    root = nodes[0]
    return root, nodes

def make_tree_unordered(V, E, R):
    """
    Factory function for creating a tree data structure of an edge-indexed array representation of a vascular
    skeleton tree instance.

    :param V:   N-by-3 vertex coordinate array. First vertex is assumed to be the root of the tree.
    :param E:   K-by-2 edge vertex index array. Edges are assumed direction going from parent to child nodes.
    :return:    A reference to the root node as well as a list of all nodes in the tree.
    """
    nodes = [None]*len(V)
    def make_node(i, p):
        nodes[i] = TreeNode()
        nodes[i].index = i
        nodes[i].position = V[i]
        nodes[i].radius = R[i]
        nodes[i].parent = nodes[p]
        for e in E[np.any(E==i, axis=1)]:
            j = e[0] if e[1] == i else e[1]
            if j != p:
                make_node(j, i)
                nodes[i].children.append(nodes[j])
    make_node(0, -1)
    root = nodes[0]
    return root, nodes

import networkx as nx
def make_tree_unordered2(V, E, R, find_root=False):
    """
    Factory function for creating a tree data structure of an edge-indexed array representation of a vascular
    skeleton tree instance.

    :param V:   N-by-3 vertex coordinate array. First vertex is assumed to be the root of the tree.
    :param E:   K-by-2 edge vertex index array. Edges are assumed direction going from parent to child nodes.
    :return:    A reference to the root node as well as a list of all nodes in the tree.
    """
    G = nx.Graph()
    for i,(v,r) in enumerate(zip(V,R)):
        G.add_node(i, v=v, r=r)
    for e in E:
        G.add_edge(e[0], e[1])

    if not nx.is_connected(G):
        print("not connected, choosing largest component...")
        idx = np.argmax([len(c) for c in nx.connected_components(G)])
        G = nx.subgraph(G, list(nx.connected_components(G))[idx]).copy()
    if not nx.is_tree(G):
        print("cycles detected, removing...")
        remove_cycles(G, len(V))
    assert(nx.is_tree(G))

    v0 = 0
    if find_root or not G.has_node(0):
        print("choosing larges radius as root...")
        v0 = [i for i in G.nodes][np.argmax([G.nodes[i]['r'] for i in G.nodes])]

    T = nx.bfs_tree(G, v0)


    nodes = [None]*len(T.nodes)
    idxs = [i for i in T.nodes]
    def make_node(i, p):
        idx_i = idxs.index(i)
        nodes[idx_i] = TreeNode()
        nodes[idx_i].index = idx_i
        nodes[idx_i].position = G.nodes[i]['v']
        nodes[idx_i].radius = G.nodes[i]['r']
        nodes[idx_i].parent = nodes[idxs.index(p)] if p != -1 else None
        for j in T.neighbors(i):
            make_node(j, i)
            nodes[idx_i].children.append(nodes[idxs.index(j)])
    make_node(v0, -1)
    root = nodes[0]
    return root, nodes

# def remove_cycles(G):
#     traversed = np.full(len(G.nodes), False)
#     def dfs(i,j):
#         if traversed[j]:
#             k = len(G.nodes)
#             G.add_node(k, v=G.nodes[j]["v"], r=G.nodes[j]["r"])
#             G.remove_edge(i,j)
#             G.add_edge(i,k)
#         else:
#             traversed[j] = True
#             for k in G.neighbors(j):
#                 if k != i:
#                     dfs(j,k)
#     traversed[0] = True
#     for j in G.neighbors(0):
#         dfs(0, j)

def remove_cycles(G, n):
    edges = list(nx.minimum_spanning_edges(G, data=False))
    # T = nx.minimum_spanning_tree(G)

    to_remove = []
    for e in G.edges:
        if e not in edges and (e[1],e[0]) not in edges:
            # i = min(e)
            # j = max(e)
            to_remove.append(e)

    k = n
    for i,j in to_remove:
        G.add_node(k, v=G.nodes[j]["v"], r=G.nodes[j]["r"])
        G.remove_edge(i,j)
        G.add_edge(i,k)
        k += 1
def limit_radius_growth(root, max_mul):
    def _limit_radius_growth_rec(node):
        for child in node.children:
            if child.radius > max_mul*node.radius:
                child.radius = max_mul*node.radius
            _limit_radius_growth_rec(child)
    _limit_radius_growth_rec(root)

def prune_tiny_leaves(root, threshold):
    def prune_tiny_leaves_rec(node):
        pruned_children = []
        for child in node.children:
            prune_tiny_leaves_rec(child)
            if len(child.children) != 0 or child.radius > threshold:
                pruned_children.append(child)
        node.children = pruned_children
    prune_tiny_leaves_rec(root)

def prune_zero_radius(root):
    def prune_zero_radius_rec(node):
        pruned_children = []
        for child in node.children:
            if child.radius > 0:
                pruned_children.append(child)
                prune_zero_radius_rec(child)
        node.children = pruned_children
    prune_zero_radius_rec(root)

def prune_tiny_offshoots(root):
    def prune_tiny_offshoots_rec(node):
        pruned_children = []
        for child in node.children:
            prune_tiny_offshoots_rec(child)
            dr = child.radius/node.radius
            if dr > 0.6 or size(child) > 1/(dr*dr*dr*3):
                pruned_children.append(child)
        node.children = pruned_children
    prune_tiny_offshoots_rec(root)

def widen_leaves(root, frac):
    def widen_leaves_rec(node, parent):
        if len(node.children) == 0:
            node.radius = frac*parent.radius + (1-frac)*node.radius
        else:
            for child in node.children:
                widen_leaves_rec(child, node)
    widen_leaves_rec(root, None)

def distfunc(pnt, start, end):
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
    diff = nearest-pnt
    # diff[3] *= 0.1
    dist = np.linalg.norm(diff)
    return dist

def simplify_edges(root, rdp_eps):
    def _simplify_edges(node, rdp_eps):
        nodes = [node.parent, node]
        while len(node.children) == 1:
            node = node.children[0]
            nodes.append(node)
        points = np.array([np.hstack((node.position, node.radius)) for node in nodes])
        # points = np.array([node.position for node in nodes])
        sz = max(np.linalg.norm(points[0, :3]-points[-1, :3]), 4*np.max(points[:,3]))
        # sz = np.linalg.norm(points[0]-points[-1])
        mask = rdp(points, rdp_eps*sz, dist=distfunc, return_mask=True)

        last = nodes[0]
        idx = last.children.index(nodes[1])
        for i, keep in enumerate(mask):
            if i == 0: continue
            if keep:
                last.children[idx] = nodes[i]
                nodes[i].parent = last
                last = nodes[i]
                idx = 0

        for child in node.children:
            _simplify_edges(child, rdp_eps)
    for child in root.children:
        _simplify_edges(child, rdp_eps)


def merge_groupings(root, eps):
    def merg(node):
        new_children = []
        def merge_rec(child):
            l = np.linalg.norm(child.position - node.position)
            r = max(node.radius, child.radius)
            if l/r > eps:
                new_children.append(child)
                child.parent = node
            else:
                for gchild in child.children:
                    merge_rec(gchild)
        for child in node.children:
            merge_rec(child)
        node.children = new_children
        for child in node.children:
            merg(child)
    merg(root)

def get_nodes(node):
    return [node] + [other for child in node.children for other in get_nodes(child)]

def make_arrays(nodes):
    V = np.empty((len(nodes), 3))
    E = []
    R = np.empty(len(nodes))
    
    for i, node in enumerate(nodes):
        V[i] = node.position
        R[i] = node.radius
        if node.parent is not None:
            j = nodes.index(node.parent)
            E.append([j, i])

    E = np.array(E)
    return V, E, R

def size(node):
    return 1 + np.sum([size(child) for child in node.children])
