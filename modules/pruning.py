from node import Node
from ID3 import *
from operator import xor

# Note, these functions are provided for your reference.  You will not be graded on their behavior,
# so you can implement them as you choose or not implement them at all if you want to use a different
# architecture for pruning.

def prune_tree(tree, nodes, validation_examples, old_acc):
    
    while reduced_by >0:
        reduction = []
        for n in nodes:
            if isinstance(n, TreeLeaf):
                nodes.pop(nodes.index(n))
                continue
            else:
                target_class = n.mode
                n.toLeaf(target_class)
                new_acc = tree_accuracy(validation_examples, tree)
                diff = new_acc - old_acc
                reduction.append(diff)
                n.toFork()
        if reduction != []:
            max_red_at = reduction.index(max(reduction))
            if isinstance(nodes[max_red_at], TreeFork):
                nodes[max_red_at].toLeaf(nodes[max_red_at].mode)
            nodes.pop(max_red_at)
            reduced_by = max(reduction)
            old_acc = tree_accuracy(validation_examples, tree)
        else:
            reduced_by = 0

    print "The new accuracy is: " + str(new_acc) + "%"
    return [tree, new_acc]


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

    # If root is a leaf, classify
    if root.label is not None:


    # Get list of children
    if root.is_nominal:
        # root.children is a dictionary
        children = [val for val in root.children.values()]
    else:
        # root.children is a pair
        children = root.children

    # Consider each node for pruning
    for node in children:
        pruned_tree = deepcopy(node)

        # Remove the subtree at that node, make it a leaf, and assign the most common class at that node
        mode_class = node.label

        leaf = Node()
        leaf.label = mode_class

        # For each subtree, we replace it with a leaf node labeled with the training instances covered by the subtree (appropriately for classification or regression).


    # Prune if the resulting tree performs no worse then the original on the validation set
    original_cci = validation_accuracy(root, validation_set)
    test_cci = validation_accuracy(test_tree, validation_set)
    if test_cci >= original_cci:
        
        # If the leaf node does not perform worse than the subtree on the pruning set, we prune the subtree and keep the leaf node because the additional complexity of the subtree is not justified; otherwise, we keep the subtree.
    
    # Nodes are removed iteratively choosing the node whose removal most increases the decision tree accuracy on the graph

    # Pruning continues until further pruning is harmful


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