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

    # If dataset empty, return None
    if not data_set:
        raise Exception("Dataset empty")

    # Always store the mode in the tree.mode field
    # This is distinct from tree.label, which should be None
    # if we end up splitting on an attribute
    tree.mode = mode(data_set)
    
    # If dataset is homogenous, tree is a leaf node
    homogenous = check_homogenous(data_set)

    if homogenous:
        tree.label = 1
        tree.value = homogenous

    # If any of the following are true, tree is a leaf with mode classification:
    # - Max depth reached
    # - Splits on all nominal attributes yield zero information gain
    # - No numerical splits left
    attr, threshold = pick_best_attribute(data_set, attribute_metadata, numerical_splits_count)

    if (depth == 0 or not attr):
        # Return ZeroR classification
        tree.label = 0
        tree.value = mode(data_set)
    
    # pick_best_attribute returned an attribute, so we split
    else:
        tree.decision_attrib = attr  # index of the attribute in attribute_metadata
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

    # pass

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

def pick_best_attribute(data_set, attribute_metadata, numerical_splits_count):
    '''
    Input:  A data_set, attribute_metadata, splits counts for numeric

    Job:    Find the attribute that maximizes the gain ratio. If attribute is numeric return best split value.
            If nominal, then split value is False.
            If gain ratio of all the attributes is 0, then return False, False
            Only consider numeric splits for which numerical_splits_count is greater than zero

    Output: index of best attribute in the attribute_metadata list, split value if numeric
    '''
    # Your code here

    # Get indices of all numerical attributes
    # numerical_attrib_indices = [i for attrib, i in enumerate(attribute_metadata) if not attrib["is_nominal"]]

    # find best attribute
    best_attri = 0
    splitValue = False
    max_gain_ratio = 0

    for attribute in range(1,len(data_set[0])):
        if attribute_metadata[attribute]['is_nominal']:
            gain_ratio = gain_ratio_nominal(data_set, attribute)
            numerical_splits_count[attribute] -= 1
        else:
            if numerical_splits_count[attribute] > 0:
                gain_ratio = gain_ratio_numeric(data_set, attribute, 1)[0]
                numerical_splits_count[attribute] -= 1
        if gain_ratio > max_gain_ratio:
            best_attri = attribute
            max_gain_ratio = gain_ratio
            if attribute_metadata[attribute]['is_nominal']:
                splitValue = False    
            else:
                splitValue = gain_ratio_numeric(data_set, attribute, 1)[1]
    if max_gain_ratio == 0:
        return False, False
    return best_attri, splitValue
    pass

# # ======== Test Cases =============================
# numerical_splits_count = [20,20]
# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
# data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [0, 0.51], [1, 0.4]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, 0.51)
# attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "weather",'is_nominal': True}]
# data_set = [[0, 0], [1, 0], [0, 2], [0, 2], [0, 3], [1, 1], [0, 4], [0, 2], [1, 2], [1, 5]]
# pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, False)

# Uses gain_ratio_nominal or gain_ratio_numeric to calculate gain ratio.

def mode(data_set):
    '''
    Input:  A data_set

    Job:    Takes a data_set and finds mode of index 0.

    Output: mode of index 0.
    '''

    counts = [0, 0]

    for x in data_set:
        categorical = x[0]
        counts[categorical] += 1

    return int(not counts[0] > counts[1])

def entropy(data_set):
    """Calculates the Shannon entropy of a dataset for the attribute at the 0th index.
    
    Arguments:
        data_set {[List of Examples]} -- list of examples given as lists of attribute values, where the 0th attribute is the classification

    Returns:
        {Float} entropy of the attribute
    """

    # Compute the probability of each outcome
    total_examples = len(data_set)
    pos_examples = len([x for x in data_set if x[0] == 1])
    neg_examples = len([x for x in data_set if x[0] == 0])

    pr_pos = float(pos_examples) / float(total_examples)
    pr_neg = float(neg_examples) / float(total_examples)

    # Compute the entropy over each event in the sample space
    entropy_pos = -pr_pos * math.log(pr_pos, 2) if pr_pos > 0.0 else 0
    entropy_neg = -pr_neg * math.log(pr_neg, 2) if pr_neg > 0.0 else 0

    return round(entropy_pos + entropy_neg, 3)


def actualEntropy(data_set):
    # Your code here
    firstNum = []
    for li in data_set:
        #print li
        firstNum.append(li[0])

    # number of unique values in the list
    unique = set(firstNum)
    probabilities = []
    for num in unique:
        probabilities.append(firstNum.count(num))
    #total occurences of numbers in the list
    total = len(firstNum)
    result = 0
    for probability in probabilities:
        probability = float(probability)/float(total)
        result += -probability * math.log(probability, 2)
    return result


def gain_ratio_nominal(data_set, attribute):
    '''
    Input:  Subset of data_set, index for a nominal attribute
    Job:    Finds the gain ratio of a nominal attribute in relation to the variable we are training on.
    Output: Returns gain_ratio. See https://en.wikipedia.org/wiki/Information_gain_ratio
    '''
    # get entropy of the data_set
    entropyVal = actualEntropy(data_set)
    # sort the data_set based on the attribute
    # [[1, 0], [1, 0], [0,0], [1,1], [1,2], [0,2], [0,2], [1,3], [0,3], [0,4]]
    total = len(data_set)
    ig = 0
    iv = 0
    sumVal = 0
    dic = split_on_nominal(data_set, attribute)
    for key in dic.keys():
        proportion = float(len(dic[key])) / float(total)
        sumVal += proportion * entropy(dic[key])
        iv += proportion * math.log(proportion, 2);
    ig = entropyVal - sumVal
    iv = -iv
    if iv == 0:
        return 0
    result = float(ig)/float(iv) 
    return result

# ======== Test case =============================
# data_set, attr = [[1, 2], [1, 0], [1, 0], [0, 2], [0, 2], [0, 0], [1, 3], [0, 4], [0, 3], [1, 1]], 1
# gain_ratio_nominal(data_set,attr) == 0.11470666361703151
# data_set, attr = [[1, 2], [1, 2], [0, 4], [0, 0], [0, 1], [0, 3], [0, 0], [0, 0], [0, 4], [0, 2]], 1
# gain_ratio_nominal(data_set,attr) == 0.2056423328155741
# data_set, attr = [[0, 3], [0, 3], [0, 3], [0, 4], [0, 4], [0, 4], [0, 0], [0, 2], [1, 4], [0, 4]], 1
# gain_ratio_nominal(data_set,attr) == 0.06409559743967516

def gain_ratio_numeric(data_set, attribute, steps):
    '''
    Input:  Subset of data set, the index for a numeric attribute, and a step size for normalizing the data.

    Job:    Calculate the gain_ratio_numeric and find the best single threshold value
            The threshold will be used to split examples into two sets
                 those with attribute value GREATER THAN OR EQUAL TO threshold
                 those with attribute value LESS THAN threshold
            Use the equation here: https://en.wikipedia.org/wiki/Information_gain_ratio
            And restrict your search for possible thresholds to examples with array index mod(step) == 0

    Output: This function returns the gain ratio and threshold value
    '''
    gain_ratio = 0
    splitValue = 0
    
    for i in range(0, len(data_set)):
        if i % steps == 0:
            threshold = data_set[i][attribute]
            tu = split_on_numerical(data_set, attribute, threshold)
            #avoid changing the original split tuple
            tu = copy.deepcopy(tu)
            for smaller in tu[0]:
                smaller[attribute] = 1
            for greater in tu[1]:
                greater[attribute] = 2
            newArr = tu[0] + tu[1]
            
            currRatio = gain_ratio_nominal(newArr, attribute) 
            #print currRatio
            if currRatio > gain_ratio:
                gain_ratio = currRatio
                splitValue = threshold
        
    return gain_ratio, splitValue
# ======== Test case =============================
# data_set,attr,step = [[1,0.05], [1,0.17], [1,0.64], [0,0.38], [1,0.19], [1,0.68], [1,0.69], [1,0.17], [1,0.4], [0,0.53]], 1, 2
# gain_ratio_numeric(data_set,attr,step) == (0.31918053332474033, 0.64)
# data_set,attr,step = [[1, 0.35], [1, 0.24], [0, 0.67], [0, 0.36], [1, 0.94], [1, 0.4], [1, 0.15], [0, 0.1], [1, 0.61], [1, 0.17]], 1, 4
# gain_ratio_numeric(data_set,attr,step) == (0.11689800358692547, 0.94)
# data_set,attr,step = [[1, 0.1], [0, 0.29], [1, 0.03], [0, 0.47], [1, 0.25], [1, 0.12], [1, 0.67], [1, 0.73], [1, 0.85], [1, 0.25]], 1, 1
# gain_ratio_numeric(data_set,attr,step) == (0.23645279766002802, 0.29)

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

    # Tests for ID3
    attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
    data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [1, 0.42], [0, 0.51], [1, 0.4]]
    numerical_splits_count = [5, 5]

    # ID3(data_set, attribute_metadata, numerical_splits_count, 10)

    print "All tests passed"