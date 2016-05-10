1. How did you handle missing attributes in examples?

    We left missing attributes alone (i.e. left them as None).
      * For nominal variables, this amounts to treating "missing" as its own attribute value, which makes sense because it's possible the "missing" nature of the value is meaningful.
      * For numerical variables, this amounts to classifying missing values as below the splitting threshold every time, since Python will always evaluate None < i for any integer i.

    We decided to handle missing attributes this way because it increased our accuracy significantly over the original, more involved implementation, which was:
      * For nominal variables, we assigned the mode attribute value across all examples at that node.
      * For numerical variables, we assigned the mean attribute value across all examples at that node.

2. Apply your algorithm to the training set, without pruning. Print out a Boolean formula in disjunctive normal form that corresponds to the unpruned tree learned from the training set. For the DNF assume that group label "1" refers to the positive examples. NOTE: if you find your tree is cumbersome to print in full, you may restrict your print-out to only 16 leaf nodes.

    ```

    ```

3. Explain in English one of the rules in this (unpruned) tree.


4. How did you implement pruning?
  
    Originally we used greedy reduced-error pruning, converting each non-leaf node to a leaf, assigning the mode classification over examples at that node, and computing the resulting accuracy on the validation set. We greedily selected the node whose pruning contributed to the greatest increase in accuracy. This process continued until the accuracy gain was zero.

    Since this was computationally intractable for the larger data set, we switched to non-greedy reduced error pruning, which pruned the first node starting from the root that contributed to an increase in accuracy.

5. Apply your algorithm to the training set, with pruning. Print out a Boolean formula in disjunctive normal form that corresponds to the pruned tree learned from the training set.

6. What is the difference in size (number of splits) between the pruned and unpruned trees?
    
    The unpruned tree has 7004 nodes. The pruned tree has 495.

7. Test the unpruned and pruned trees on the validation set. What are the accuracies of each tree? Explain the difference, if any.

8. Create learning curve graphs for both unpruned and pruned trees. Is there a difference between the two graphs?

9. Which tree do you think will perform better on the unlabeled test set? Why? Run this tree on the test file and submit your predictions as described in the submission instructions.

10. Which members of the group worked on which parts of the assignment?