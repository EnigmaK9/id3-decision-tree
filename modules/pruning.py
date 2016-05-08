from node import Node
from ID3 import *
from operator import xor
from copy import deepcopy

# Note, these functions are provided for your reference.  You will not be graded on their behavior,
# so you can implement them as you choose or not implement them at all if you want to use a different
# architecture for pruning.

def reduced_error_pruning(root, training_set, validation_set):
    """Reduced error pruning for a trained ID3 decision tree.
    
    Removes nodes such that doing so improves validation accuracy.
    
    Arguments:
        root {Node} -- root node of decision tree learned over the training set
        training_set {List of Examples} -- list of examples given as lists of attribute values
        validation_set {List of Examples} -- list of examples disjoint from the training set, given as lists of attribute values

    Returns:
        Node -- the improved root of the pruned decision tree
    """

    # If we're at a leaf node, just return the leaf
    if root.label is not None:
        return root

    # If root has children, prune children
    if root.children:
        if root.is_nominal:
            children = root.children.values()
        else:
            children = root.children

        for c in children:
            reduced_error_pruning(child, training_set, validation_set)

    accuracy = validation_accuracy(root, validation_set)
    
    root.make_leaf()
    new_accuracy = validation_accuracy(root, validation_set)

    if new_accuracy < accuracy:
        c.make_fork()

    return root


def validation_accuracy(tree,validation_set):
    """Takes a tree and a validation set, and returns the accuracy of the set on the given tree in terms of correctly classified instances.
    
    Arguments:
        tree {Node} -- the root of the decision tree learned over a set of training data
        validation_set {List of Examples} -- list of examples given as sequences of attribute values
    
    Returns:
        {Float} -- percentage of correctly classified instances
    """

    total_instances = len(validation_set)
    accurate_instances = 0

    for x in validation_set:
        true_class = x[0]  # winner classification
        computed_class = tree.classify(x)
        if computed_class == true_class:
            accurate_instances += 1

    return accurate_instances / float(total_instances)