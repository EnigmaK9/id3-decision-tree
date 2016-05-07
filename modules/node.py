from ID3 import *
from pprint import pprint

# DOCUMENTATION
# =====================================
# Class node attributes:
# ----------------------------
# children - a list of 2 if numeric and a dictionary if nominal.  
#            For numeric, the 0 index holds examples < the splitting_value, the 
#            index holds examples >= the splitting value
#
# label - is the output label (0 or 1) if there are no other attributes to split on, or the data is homogenous
#         if there is a decision attribute, use mode
#
# decision_attribute - the index of the decision attribute being split on
#
# is_nominal - is the decision attribute nominal
#
# value - Ignore (not used, output class if any goes in label)
#
# splitting_value - if numeric, where to split
#
# name - name of the attribute being split on

class Node:
    def __init__(self):
        # initialize all attributes
        self.label = None
        self.decision_attribute = None
        self.is_nominal = None
        self.value = None
        self.mode = None
        self.splitting_value = None
        self.children = {}
        self.name = None

    def __str__(self):
        return str(self.label)

    def classify(self, instance):
        '''
        given a single observation, will return the output of the tree
        '''

        if type(self.label) is int:
            return self.label

        else:
            attr = self.decision_attribute
            instance_val = instance[attr]

            try:
                if self.is_nominal:
                    next = self.children["instance_val"]
                else:
                    # Node is numerical
                    if instance_val < self.splitting_value:
                        next = self.children[0]
                    else:
                        # Instance value is >= splitting value
                        next = self.children[1]

                return next.classify(instance)

            except (IndexError, KeyError):
                return self.mode

            


    def print_tree(self, indent = 0):
        '''
        returns a string of the entire tree in human readable form
        '''

        pre = "|    " * indent

        # If node is a leaf, simply print output class
        if self.label is not None:
            print pre + "CLASS: " + str(self.label)

        # Otherwise, print attribute split information
        else:
            if self.is_nominal:
                for val, child in self.children.items():
                    print "%s%s: %s" % (pre, self.name, val)
                    child.print_tree()
            else:
                # Node is numerical
                print "%s%s < %f" % (pre, self.name, self.splitting_value)
                self.children[0].print_tree(indent + 1)
                print "%s%s >= %f" % (pre, self.name, self.splitting_value)
                self.children[1].print_tree(indent + 1)
        


    def print_dnf_tree(self):
        '''
        returns the disjunct normalized form of the tree.
        '''
        pass


if __name__ == "__main__":
    import os; path = "/Users/sarah/git/id3-decision-tree/modules"; os.chdir(path)
    # attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
    # data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [1, 0.42], [0, 0.51], [1, 0.4]]
    # numerical_splits_count = [5, 5]

    # test = ID3(data_set, attribute_metadata, numerical_splits_count, 10)
    # print "Output tree:"
    # test.print_tree()

    attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
    data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [1, 0.42], [0, 0.51], [1, 0.4]]
    numerical_splits_count = [1, 1]
    n = ID3(data_set, attribute_metadata, numerical_splits_count, 5)

    print [n.classify(x) == x[0] for x in data_set]

    # pprint(vars(test))