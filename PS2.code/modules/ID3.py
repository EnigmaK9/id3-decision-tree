import math
from node import Node
import sys

def ID3(data_set, attribute_metadata, numerical_splits_count, depth):
    '''
    See Textbook for algorithm.
    Make sure to handle unknown values, some suggested approaches were
    given in lecture.

    Input:  A data_set, attribute_metadata, maximum number of splits to consider for numerical attributes,
    maximum depth to search to (depth = 0 indicates that this node should output a label)

    Output: The node representing the decision tree learned over the given data set

    Recursively chooses the most significant attribute as the root of the subtree

    TODO:
    - Handle missing attributes

    '''

    tree = Node()
    
    # If dataset empty, return None
    if not data_set:
        return None
    
    # If dataset is homogenous, return the classification
    elif check_homogenous(data_set):
        tree.label = check_homogenous(data_set)

    # Return majority classification if any of the following are true:
    # 
    # - Max tree depth reached
    # - Splits on all nominal attributes yield zero information gain
    # - No numerical splits left

    attr, threshold = pick_best_attribute(data_set, attribute_metadata, numerical_splits_count)

    numerical_attrib_indices = [i for attrib, i in enumerate(attribute_metadata) if not attrib["is_nominal"]]  # gets indices of all numerical attributes
    splits_remaining = sum([numerical_splits_count[i] for i in numerical_attrib_indices])

    if (depth == 0 or
        not attr or
        splits_remaining == 0):
        # Return ZeroR classification
        tree.label = mode(data_set)
    
    else:
        # pick_best_attribute returned an attribute, so we split
        tree.decision_attrib = attr  # index of the attribute in attribute_metadata
        tree.name = attribute_metadata[attr]["name"]
        tree.is_nominal = attribute_metadata[attr]["is_nominal"]
        tree.splitting_value = threshold  # Will be False if nominal

        if tree.is_nominal:
            # children is a dictionary of pairs (value, Node)
            children = {}

            partition = split_on_nominal(data_set, attr)

            for value, examples in partition:
                children[value] = ID3(examples, attribute_metadata, numerical_splits_count, depth - 1)

        else:
            # children is a list of two nodes [< t, >= t] for split threshold t
            children = []

            partition_below, partition_above = split_on_numerical(data_set, attr, threshold)

            # attribute is numeric, decrement its index in numerical_splits_count
            new_splits_count = numerical_splits_count
            new_splits_count[attr] -= 1

            children[0] = ID3(partition_below, attribute_metadata, new_splits_count, depth - 1)
            children[1] = ID3(partition_above, attribute_metadata, new_splits_count, depth - 1)

        # Set the children property of the node, whether attribute was numeric or nominal
        tree.children = children


    return tree

    # pass

def check_homogenous(data_set):
    '''
    Input:  A data_set

    Job:    Checks if the attribute at index 0 is the same for the data_set, if so return output otherwise None.

    Output: Return either the homogenous attribute or None
    '''
    # Your code here
    pass
    
# ======== Test Cases =============================
# data_set = [[0],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  None
# data_set = [[0],[1],[None],[0]]
# check_homogenous(data_set) ==  None
# data_set = [[1],[1],[1],[1],[1],[1]]
# check_homogenous(data_set) ==  1

def pick_best_attribute(data_set, attribute_metadata, numerical_splits_count):
    '''
    ========================================================================================================
    Input:  A data_set, attribute_metadata, splits counts for numeric
    ========================================================================================================
    Job:    Find the attribute that maximizes the gain ratio. If attribute is numeric return best split value.
            If nominal, then split value is False.
            If gain ratio of all the attributes is 0, then return False, False
            Only consider numeric splits for which numerical_splits_count is greater than zero
    ========================================================================================================
    Output: index of best attribute in the attribute_metadata list, split value if numeric
    ========================================================================================================
    '''
    # Your code here
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

def mode_tests():
    data_set = [[0],[1],[1],[1],[1],[1]]
    assert(mode(data_set) == 1)
    data_set = [[0],[1],[0],[0]]
    assert(mode(data_set) == 0)

def entropy(data_set):
    '''
    ========================================================================================================
    Input:  A data_set
    ========================================================================================================
    Job:    Calculates the entropy of the attribute at the 0th index, the value we want to predict.
    ========================================================================================================
    Output: Returns entropy. See Textbook for formula
    ========================================================================================================
    '''


# ======== Test case =============================
# data_set = [[0],[1],[1],[1],[0],[1],[1],[1]]
# entropy(data_set) == 0.811
# data_set = [[0],[0],[1],[1],[0],[1],[1],[0]]
# entropy(data_set) == 1.0
# data_set = [[0],[0],[0],[0],[0],[0],[0],[0]]
# entropy(data_set) == 0


def gain_ratio_nominal(data_set, attribute):
    '''
    ========================================================================================================
    Input:  Subset of data_set, index for a nominal attribute
    ========================================================================================================
    Job:    Finds the gain ratio of a nominal attribute in relation to the variable we are training on.
    ========================================================================================================
    Output: Returns gain_ratio. See https://en.wikipedia.org/wiki/Information_gain_ratio
    ========================================================================================================
    '''
    # Your code here
    pass
# ======== Test case =============================
# data_set, attr = [[1, 2], [1, 0], [1, 0], [0, 2], [0, 2], [0, 0], [1, 3], [0, 4], [0, 3], [1, 1]], 1
# gain_ratio_nominal(data_set,attr) == 0.11470666361703151
# data_set, attr = [[1, 2], [1, 2], [0, 4], [0, 0], [0, 1], [0, 3], [0, 0], [0, 0], [0, 4], [0, 2]], 1
# gain_ratio_nominal(data_set,attr) == 0.2056423328155741
# data_set, attr = [[0, 3], [0, 3], [0, 3], [0, 4], [0, 4], [0, 4], [0, 0], [0, 2], [1, 4], [0, 4]], 1
# gain_ratio_nominal(data_set,attr) == 0.06409559743967516

def gain_ratio_numeric(data_set, attribute, steps):
    '''
    ========================================================================================================
    Input:  Subset of data set, the index for a numeric attribute, and a step size for normalizing the data.
    ========================================================================================================
    Job:    Calculate the gain_ratio_numeric and find the best single threshold value
            The threshold will be used to split examples into two sets
                 those with attribute value GREATER THAN OR EQUAL TO threshold
                 those with attribute value LESS THAN threshold
            Use the equation here: https://en.wikipedia.org/wiki/Information_gain_ratio
            And restrict your search for possible thresholds to examples with array index mod(step) == 0
    ========================================================================================================
    Output: This function returns the gain ratio and threshold value
    ========================================================================================================
    '''
    # Your code here
    pass
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

def split_on_nominal_tests():
    data_set, attr = [[0, 4], [1, 3], [1, 2], [0, 0], [0, 0], [0, 4], [1, 4], [0, 2], [1, 2], [0, 1]], 1
    assert(split_on_nominal(data_set, attr) == {0: [[0, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3]], 4: [[0, 4], [0, 4], [1, 4]]})
    data_set, attr = [[1, 2], [1, 0], [0, 0], [1, 3], [0, 2], [0, 3], [0, 4], [0, 4], [1, 2], [0, 1]], 1
    assert(split_on_nominal(data_set, attr) == {0: [[1, 0], [0, 0]], 1: [[0, 1]], 2: [[1, 2], [0, 2], [1, 2]], 3: [[1, 3], [0, 3]], 4: [[0, 4], [0, 4]]})

def split_on_numerical(data_set, attribute, splitting_value):
    '''
    Input: Subset of data set, the index for a numeric attribute, threshold (splitting) value

    Job: Splits data_set into a tuple of two lists,
            - the first list contains the examples where the given attribute has value less than the splitting value,
            - the second list contains the other examples.

    Output: Tuple of two lists as described above
    '''

    examples_below = [x for x in data_set if x[attr] < splitting_value]
    examples_above = [x for x in data_set if x[attr] >= splitting_value]

    return examples_below, examples_above
    
def split_on_numerical_tests():
    d_set,a,sval = [[1, 0.25], [1, 0.89], [0, 0.93], [0, 0.48], [1, 0.19], [1, 0.49], [0, 0.6], [0, 0.6], [1, 0.34], [1, 0.19]],1,0.48
    assert(split_on_numerical(d_set,a,sval) == ([[1, 0.25], [1, 0.19], [1, 0.34], [1, 0.19]],[[1, 0.89], [0, 0.93], [0, 0.48], [1, 0.49], [0, 0.6], [0, 0.6]]))
    d_set,a,sval = [[0, 0.91], [0, 0.84], [1, 0.82], [1, 0.07], [0, 0.82],[0, 0.59], [0, 0.87], [0, 0.17], [1, 0.05], [1, 0.76]],1,0.17
    assert(split_on_numerical(d_set,a,sval) == ([[1, 0.07], [1, 0.05]],[[0, 0.91],[0, 0.84], [1, 0.82], [0, 0.82], [0, 0.59], [0, 0.87], [0, 0.17], [1, 0.76]]))

if __name__ == "__main__":
    mode_tests()
    split_on_nominal_tests()
    split_on_numerical_tests()
