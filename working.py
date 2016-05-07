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

    # Get indices of all numerical attributes
    # numerical_attrib_indices = [i for attrib, i in enumerate(attribute_metadata) if not attrib["is_nominal"]]

    result = False, False
    igr_max = 0

    # Only consider attributes whose numerical_splits_count value is > 0
    for i, a in enumerate(attribute_metadata):
        if a["is_nominal"]:
            igr = gain_ratio_nominal(data_set, i)
        else:
            # Attribute is numerical, check if splits remaining
            if numerical_splits_count[i] > 0:
                igr, threshold = gain_ratio_numeric(data_set, i)
        
        if igr > igr_max:
            igr_max = igr  # Update internal counter
            result = i, threshold

    return result


if __name__ == "__main__":
    numerical_splits_count = [20,20]
    attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "opprundifferential",'is_nominal': False}]
    data_set = [[1, 0.27], [0, 0.42], [0, 0.86], [0, 0.68], [0, 0.04], [1, 0.01], [1, 0.33], [1, 0.42], [0, 0.51], [1, 0.4]]
    assert pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, 0.51)
    attribute_metadata = [{'name': "winner",'is_nominal': True},{'name': "weather",'is_nominal': True}]
    data_set = [[0, 0], [1, 0], [0, 2], [0, 2], [0, 3], [1, 1], [0, 4], [0, 2], [1, 2], [1, 5]]
    assert pick_best_attribute(data_set, attribute_metadata, numerical_splits_count) == (1, False)

    # Uses gain_ratio_nominal or gain_ratio_numeric to calculate gain ratio.