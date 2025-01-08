import sys
import os
sys.path.append(os.path.abspath('../'))
import tree as TREE
import numpy as np
"""
Ensures cardinality of tree. Designed to work on vein tree alone,
so run before generating arterial dual.
"""

def split_vertex(nodes, node):
    d = node.position - node.parent.position
    d = d/np.linalg.norm(d)
    r = node.radius

    new_node = TREE.TreeNode()
    new_node.index = len(nodes)
    new_node.position = node.position.copy()
    new_node.radius = r
    new_node.parent = node

    kept_children = int(np.floor(len(node.children)/2))
    new_node.children = node.children[kept_children:]
    for child in new_node.children:
        child.parent = new_node
    node.children = node.children[:kept_children]
    node.children.append(new_node)

    node.position -= r*d 
    new_node.position += r*d 

    nodes.append(new_node)

    return nodes

def ensure_cardinality(V, E, R):
    _, nodes = TREE.make_tree(V, E, R)
    i = 0
    while i < len(nodes):
        node = nodes[i]
        while len(node.children) > 2:
            assert(node.parent is not None) # Root of tree is not ternary.
            nodes = split_vertex(nodes, node)
        i += 1
    return TREE.make_arrays(nodes)
