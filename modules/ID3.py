import math
from node import Node
import sys
import copy

def ID3(data_set, attribute_metadata, numerical_splits_count, depth):
    """Recursively trains a binary classification decision tree, using Quinlan's ID3 algorithm.

    Recursively chooses the most significant attribute as the root of the subtree.
    
    Arguments:
        data_set {List of Examples} -- a two-dimensional list of examples, where each example is a sequence of attribute values
        attribute_metadata {List of Dicts} -- a list of attribute dictionaries, each with a name and type field
        numerical_splits_count {List of Ints} -- a list of integers ordered corresponding to the attribute_metadata order, representing the number of remaining splits allowed for each numerical attribute
        depth {Int} -- maximum depth to search to (depth = 0 indicates that this node should output a label)
    
    Returns:
        Node -- the root of the decision tree learned over the given data set
    
    Raises:
        Exception -- if given dataset is empty
    """

    tree = Node()
    tree.mode = mode(data_set)
    
    # If dataset is homogenous, tree is a leaf node
    homogenous = check_homogenous(data_set)

    if homogenous:
        tree.label = homogenous

    # If any of the following are true, tree is a leaf with mode classification:
    # - Max depth reached
    # - Splits on all nominal attributes yield zero information gain
    # - No numerical splits left
    # Return mode classification
    attr, threshold = pick_best_attribute(data_set, attribute_metadata, numerical_splits_count)

    if (depth == 0 or not attr):
        tree.label = mode(data_set)
    
    # Otherwise, pick_best_attribute returned an attribute, so we split
    else:
        tree.decision_attrib = attr  # index of the attribute
        tree.name = attribute_metadata[attr]["name"]
        tree.is_nominal = attribute_metadata[attr]["is_nominal"]
        tree.splitting_value = threshold  # Will be False if nominal

        # Handle splitting on a nominal attribute
        if tree.is_nominal:
            # children is a dictionary of pairs (value, Node)
            children = {}

            partition = split_on_nominal(data_set, attr)

            for value, examples in partition:
                children[value] = ID3(examples, attribute_metadata, numerical_splits_count, depth - 1)

        # Handle splitting on a numerical attribute
        else:
            # children is a list of two nodes [< t, >= t] for split threshold t
            children = []

            partition_below, partition_above = split_on_numerical(data_set, attr, threshold)

            # Attribute is numeric, so we decrement its index in numerical_splits_count
            new_splits_count = numerical_splits_count
            new_splits_count[attr] -= 1

            children.append(ID3(partition_below, attribute_metadata, new_splits_count, depth - 1))
            children.append(ID3(partition_above, attribute_metadata, new_splits_count, depth - 1))

        # Set the children property of the node, whether attribute was numeric or nominal
        tree.children = children

    return tree



def check_homogenous(data_set):
    """Checks if a dataset has a homogenous output classification.
    
    Checks if the attribute at index 0 for each example is the same for all examples in the data set.
    
    Arguments:
        data_set {List of Examples} -- list of examples given as lists of attribute values

    Returns:
        either the homogenous attribute or None
    """

    # Get the classification of the first example
    value = data_set[0][0]
    
    for x in data_set[1:]:
        if x[0] != value:
            return None

    return value


def mode(data_set):
    """Returns the mode of index 0 for a given dataset.
    
    Arguments:
        data_set {List of Examples}
    
    Returns:
        {Int} -- 0 or 1, the mode of index 0
    """

    counts = [0, 0]

    for x in data_set:
        categorical = x[0]
        counts[categorical] += 1

    return int(not counts[0] > counts[1])


def entropy(data_set):
    """Calculates the Shannon entropy of a dataset for the binary attribute at the 0th index.
    
    Arguments:
        data_set {[List of Examples]} -- list of examples given as lists of attribute values, where the 0th attribute is the classification

    Returns:
        {Float} entropy of the attribute
    """

    # Compute the probability of each outcome
    total_examples = len(data_set)
    pos_examples = len([x for x in data_set if x[0] == 1])
    neg_examples = len([x for x in data_set if x[0] == 0])

    # Catch empty partitions
    if total_examples == 0 or pos_examples == 0:
        pr_pos = 0
    else:
        pr_pos = float(pos_examples) / float(total_examples)

    if total_examples == 0 or neg_examples == 0:
        pr_neg = 0
    else:
        pr_neg = float(neg_examples) / float(total_examples)

    # Compute the entropy over each event in the sample space
    entropy_pos = -pr_pos * math.log(pr_pos, 2) if pr_pos > 0.0 else 0
    entropy_neg = -pr_neg * math.log(pr_neg, 2) if pr_neg > 0.0 else 0

    return entropy_pos + entropy_neg


def split_on_nominal(data_set, attribute):
    '''
    Input:  subset of data set, the index for a nominal attribute.
    Job:    Creates a dictionary of all values of the attribute.
    Output: Dictionary of all values pointing to a list of all the data with that attribute
    '''
    # partition is a dictionary of pairs (value, subset)
    partition = {}

    # Get range of possible attribute values
    values = {x[attribute] for x in data_set}

    for value in values:
        partition[value] = [x for x in data_set if x[attribute] == value]

    return partition


def split_on_numerical(data_set, attribute, splitting_value):
    '''
    Input: Subset of data set, the index for a numeric attribute, threshold (splitting) value

    Job: Splits data_set into a tuple of two lists,
            - the first list contains the examples where the given attribute has value less than the splitting value,
            - the second list contains the other examples.

    Output: Tuple of two lists as described above
    '''

    examples_below = [x for x in data_set if x[attribute] < splitting_value]
    examples_above = [x for x in data_set if x[attribute] >= splitting_value]

    return examples_below, examples_above


def gain_ratio_nominal(data_set, attribute):
    """Compute the information gain ratio for splitting over a nominal variable.
    
    Arguments:
        data_set {List of Examples}
        attribute {Int} -- index of a nominal attribute

    Returns:
        {Float} -- the gain ratio
    """
    
    current_entropy = entropy(data_set)
    total_examples = len(data_set)
    partition = split_on_nominal(data_set, attribute)
    
    # Compute info gain and intrinsic value of test over all attribute values
    entropy_after = 0
    intrinsic_value = 0

    for subset in partition.values():
        # If split is pure, partial IG and IV will be 0
        if total_examples > 0:
            p = len(subset) / float(total_examples)  # Probability an example will have this this attribute value
            entropy_after += p * entropy(subset)

            # Compute partial intrinsic value
            iv = -p * math.log(p, 2)
            intrinsic_value += iv

    info_gain = current_entropy - entropy_after
    igr = info_gain / float(intrinsic_value) if intrinsic_value > 0 else 0

    return igr


def gain_ratio_numeric(data_set, attribute, steps=1):
    """Compute the gain ratio for splitting on a numeric attribute.
    
    Finds the threshold value yielding the highest gain ratio.
    Threshold value partitions the data_set into two subsets, those with attribute value < threshold, and those >= threshold
    Checks every ith example in the data_set, where i denotes the step value
    
    Arguments:
        data_set {List of Examples}
        attribute {Int} -- index of numeric attribute in an example
    
    Keyword Arguments:
        steps {Int} -- denotes the number of values to skip when iterating through data_set (default: {1})

    Returns:
        {Float, Float} -- gain ratio and threshold value
    """

    current_entropy = entropy(data_set)
    total_examples = len(data_set)

    igr_max = 0, 0

    # Iterate through sorted dataset, trying every step-th value
    for i in xrange(0, total_examples, steps):
        threshold = data_set[i][attribute]
        partition = split_on_numerical(data_set, attribute, threshold)

        # Compute the information gain at this threshold
        entropy_after = 0
        intrinsic_value = 0

        for subset in partition:
            # If the split is pure, partial IV and IG will be 0
            if len(subset) > 0:
                p = len(subset) / float(total_examples)  # Probability an example will have this this attribute value
                entropy_after += p * entropy(subset)

                # Compute partial intrinsic value
                iv = -p * math.log(p, 2)
                intrinsic_value += iv

            info_gain = current_entropy - entropy_after
            igr = info_gain / float(intrinsic_value) if intrinsic_value > 0 else 0

        # Update max if necessary
        if igr > igr_max[0]:
            igr_max = igr, threshold
    
    return igr_max


def pick_best_attribute(data_set, attribute_metadata, numerical_splits_count):
    """Picks the attribute that maximizes information gain ratio.
    
    If attribute is numeric, return best split value.
    If attribute is nominal, split value is False.
    If gain ratio of all the attributes is 0, then return False, False.
    Only consider numeric splits for which numerical_splits_count is greater than zero.
    
    Arguments:
        data_set {List of Examples}
        attribute_metadata {List of Dictionaries}
        numerical_splits_count {List of Ints} -- remaining splits for numeric variables
    
    Returns: one of the following
        {Int, Float} -- index of best attribute in the attribute_metadata list, split threshold if attribute is numeric
        {Int, False} -- index of best attribute in the attribute_metadata list, False if attribute is nominal
        {False, False} -- if no best attribute exists
    """

    result = False, False
    igr_max = 0

    # Only consider attributes whose numerical_splits_count value is > 0
    # Enumerate from attribute_metadata[1:] to skip the classification attribute
    for i, a in enumerate(attribute_metadata[1:]):
        i += 1  # Reset i to account for skipping classification attribute
        if a["is_nominal"]:
            igr = gain_ratio_nominal(data_set, i)
            t = False
        elif numerical_splits_count[i] > 0:
            igr, t = gain_ratio_numeric(data_set, i, steps=1)
        
        if igr > igr_max:
            igr_max = igr  # Update internal counter
            result = i, t

    return result


if __name__ == "__main__":
    # Tests for check_homogenous
    data_set = [[0],[1],[1],[1],[1],[1]]
    assert(check_homogenous(data_set) == None)
    data_set = [[0],[1],[None],[0]]
    assert(check_homogenous(data_set) == None)
    data_set = [[1],[1],[1],[1],[1],[1]]
    assert(check_homogenous(data_set) == 1)
    print "Passed tests for check_homogenous"

    # Tests for mode
    data_set = [[0],[1],[1],[1],[1],[1]]
    assert(mode(data_set) == 1)
    data_set = [[0],[1],[0],[0]]
    assert(mode(data_set) == 0)
    print "Passed tests for mode"

    # Tests for split_on_nominal
    data_set, attr = [[0, 4], [1, 3], [1, 2], [0, 0], [0, 0], [0, 4], [1, 4], [0, 2], [1, 2], [0, 1]], 1
    assert(split_on_nominal(data_set, attr) == {0: [[0, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3]], 4: [[0, 4], [0, 4], [1, 4]]})
    data_set, attr = [[1, 2], [1, 0], [0, 0], [1, 3], [0, 2], [0, 3], [0, 4], [0, 4], [1, 2], [0, 1]], 1
    assert(split_on_nominal(data_set, attr) == {0: [[1, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3], [0, 3]], 4: [[0, 4], [0, 4]]})
    print "Passed tests for split_on_nominal"

    # Tests for split_on_numerical
    d_set,a,sval = [[1, 0.25], [1, 0.89], [0, 0.93], [0, 0.48], [1, 0.19], [1, 0.49], [0, 0.6], [0, 0.6], [1, 0.34], [1, 0.19]],1,0.48
    assert(split_on_numerical(d_set,a,sval) == ([[1, 0.25], [1, 0.19], [1, 0.34], [1, 0.19]],[[1, 0.89], [0, 0.93], [0, 0.48], [1, 0.49], [0, 0.6], [0, 0.6]]))
    d_set,a,sval = [[0, 0.91], [0, 0.84], [1, 0.82], [1, 0.07], [0, 0.82],[0, 0.59], [0, 0.87], [0, 0.17], [1, 0.05], [1, 0.76]],1,0.17
    assert(split_on_numerical(d_set,a,sval) == ([[1, 0.07], [1, 0.05]],[[0, 0.91],[0, 0.84], [1, 0.82], [0, 0.82], [0, 0.59], [0, 0.87], [0, 0.17], [1, 0.76]]))
    print "Passed tests for split_on_numerical"

    # Tests for entropy
    data_set = [[0],[1],[1],[1],[0],[1],[1],[1]]
    assert round(entropy(data_set), 3) == 0.811
    data_set = [[0],[0],[1],[1],[0],[1],[1],[0]]
    assert entropy(data_set) == 1.0
    data_set = [[0],[0],[0],[0],[0],[0],[0],[0]]
    assert entropy(data_set) == 0
    print "Passed tests for entropy"

    # Tests for gain_ratio_nominal
    data_set, attr = [[1, 2], [1, 0], [1, 0], [0, 2], [0, 2], [0, 0], [1, 3], [0, 4], [0, 3], [1, 1]], 1
    assert gain_ratio_nominal(data_set,attr) == 0.11470666361703151
    data_set, attr = [[1, 2], [1, 2], [0, 4], [0, 0], [0, 1], [0, 3], [0, 0], [0, 0], [0, 4], [0, 2]], 1
    assert gain_ratio_nominal(data_set,attr) == 0.2056423328155741
    data_set, attr = [[0, 3], [0, 3], [0, 3], [0, 4], [0, 4], [0, 4], [0, 0], [0, 2], [1, 4], [0, 4]], 1
    assert gain_ratio_nominal(data_set,attr) == 0.06409559743967516
    print "Passed tests for gain_ratio_nominal"

    # Tests for gain_ratio_numeric
    data_set,attr,step = [[0,0.05], [1,0.17], [1,0.64], [0,0.38], [0,0.19], [1,0.68], [1,0.69], [1,0.17], [1,0.4], [0,0.53]], 1, 2
    assert gain_ratio_numeric(data_set,attr,step) == (0.31918053332474033, 0.64)
    data_set,attr,step = [[1, 0.35], [1, 0.24], [0, 0.67], [0, 0.36], [1, 0.94], [1, 0.4], [1, 0.15], [0, 0.1], [1, 0.61], [1, 0.17]], 1, 4
    assert gain_ratio_numeric(data_set,attr,step) == (0.11689800358692547, 0.94)
    data_set,attr,step = [[1, 0.1], [0, 0.29], [1, 0.03], [0, 0.47], [1, 0.25], [1, 0.12], [1, 0.67], [1, 0.73], [1, 0.85], [1, 0.25]], 1, 1
    assert gain_ratio_numeric(data_set,attr,step) == (0.23645279766002802, 0.29)
    print "Passed tests for gain_ratio_numeric"

    # Tests for pick_best_attribute
    numerical_splits_count = [20,20]
    attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
    data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [0, 0.51], [1, 0.4]]
    assert pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, 0.51)
    attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "weather",'is_nominal': True}]
    data_set = [[0, 0], [1, 0], [0, 2], [0, 2], [0, 3], [1, 1], [0, 4], [0, 2], [1, 2], [1, 5]]
    assert pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, False)
    print "Passed tests for pick_best_attribute"

    # Tests for ID3
    attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
    data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [1, 0.42], [0, 0.51], [1, 0.4]]
    numerical_splits_count = [5, 5]

    test = ID3(data_set, attribute_metadata, numerical_splits_count, 10)
    print "\n"
    test.print_tree()
    # print "Passed tests for ID3"

    print "All tests passed"