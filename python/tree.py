import numpy as np

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

def make_arrays(nodes):
    V = np.empty((len(nodes), 3))
    E = []
    R = np.empty(len(nodes))
    
    for node in nodes:
        V[node.index] = node.position
        R[node.index] = node.radius
        if node.parent is not None:
            E.append([node.parent.index, node.index])

    E = np.array(E)
    return V, E, R
