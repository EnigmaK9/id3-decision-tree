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

    # Generate a list of nodes
    nodes = [root]
    for n in nodes:
        if n.children:
            if n.is_nominal:
                nodes.extend(x for x in n.children.values())
            else:
                nodes.extend(x for x in n.children)

    if len(set(nodes)) != len(nodes):
        raise Exception("you literally fucked up BFS")

    # Compute the original accuracy for comparison
    accuracy = validation_accuracy(root, validation_set)
    print "Original accuracy: " + str(accuracy)

    # Initialize gain to accuracy to simulate do-while
    gain = accuracy

    while gain > 0:
        performance = []

        for n in nodes:
            if n.label is not None:
                # If leaf, skip node
                performance.append(None)
            else:
                # Temporarily make it a leaf to compute validation accuracy
                n.make_leaf()
                acc = validation_accuracy(root, validation_set)
                performance.append(acc)
                n.make_fork()

        i, max_performance = max(enumerate(performance), key=lambda x: x[1])
        if max_performance >= accuracy:
            gain = max_performance - accuracy
            accuracy = max_performance
            print "Increased accuracy by %f" % (gain)

            # Permanently make the node a leaf
            nodes[i].make_leaf()

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

    accuracy = accurate_instances / float(total_instances)
    return accuracy


if __name__ == "__main__":
    attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
    data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [1, 0.42], [0, 0.51], [1, 0.4]]
    numerical_splits_count = [5, 5]
    n = ID3(data_set, attribute_metadata, numerical_splits_count, 5)
    res = [n.classify(x) == x[0] for x in data_set]
    assert validation_accuracy(n, data_set) == 10 / float(11)

    # Tests for ID3
    attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
    data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [1, 0.42], [0, 0.51], [1, 0.4]]
    numerical_splits_count = [5, 5]

    test = ID3(data_set, attribute_metadata, numerical_splits_count, 10)
    print "\n"
    print test.print_tree()
    print reduced_error_pruning(test)