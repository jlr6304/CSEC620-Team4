
import numpy as np
from math import inf

class Node:
    """
    Class implemented to create a decision tree
    """
    def __init__(self, data, labels, feature_subcount, depth=0, parent_imp = inf, params = {"max_depth":10, "min_node":10}):
        """
        Node initialization
        """
        self.left = None    # left children
        self.right = None   # right children

        self.data = data        # samples in the node
        self.labels = labels    # labels of the samples

        self.depth = depth  # depth-level of the node 

        self.parentImpurity = parent_imp            # parent-node impurity
        self.impurity = gini_impurity(self.labels)  # node impurity

        self.split_feature = None   # feature on which to split the node (if not leaf)
        self.split_threshold = None # threshold on the feature on which to split the node (if not leaf)

        self.params = params  # tree parameters
        
        # --- Try to split the node 
        self.split(feature_subcount)


    def split(self, feature_subcount):
        """
        #### Method to split the node
            splits the node if possible and create and split children nodes (recursive function) 

        Parameters:
            feature_subcount: number of features on which to try to split
        """
        if not self.isLeaf(): # Split if the node isn't a leaf
            
            n_samples = self.data.shape[0]
            n_features = self.data.shape[1]

            # -- Choose randomly a subset of the features
            features = np.unique(np.random.choice(n_features, feature_subcount, replace=True))
            
            # -- Compute the best threshold (and corresponding Gini impurity) for each feature
            impurity = {f: 0 for f in features}
            splits = {f: 0 for f in features}
            for f in features:
                impurity[f], splits[f] = best_split_impurity(self.data[:, f], self.labels)
            
            # find the feature corresponding to the minimum Gini impurity
            best_f = min(impurity, key=impurity.get)
            best_s = splits[best_f]

            self.split_feature = best_f
            self.split_threshold = best_s

            # split on the feature with the lowest Gini impurity
            target_samples = self.data[:, best_f]

            mask = target_samples <= best_s

            # -- Create child nodes
            left_sub_samples = self.data[mask, :]
            left_sub_labels = self.labels[mask]
            
            right_sub_samples = self.data[np.logical_not(mask), :]
            right_sub_labels = self.labels[np.logical_not(mask)]

            # test if conditions on the nodes are satisfied 
            if len(left_sub_samples)>0 and len(right_sub_samples)>0 and self.impurity<self.parentImpurity:

                self.left = Node(left_sub_samples, left_sub_labels, feature_subcount,\
                     depth=self.depth + 1, parent_imp = self.impurity, params=self.params)

                self.right = Node(right_sub_samples, right_sub_labels, feature_subcount,\
                    depth=self.depth + 1, parent_imp = self.impurity, params=self.params)


    def isLeaf(self):
        """
        #### Test if the node is a leaf:
               returns the result as a boolean
        """
        # check if the branch has reached the max_depth
        cond1 = self.depth > self.params['max_depth']

        # check if the number of samples in the group to split is less than the min_node
        cond2 = len(self.labels) < self.params['min_node']        

        return cond1 or cond2

    def predict(self, sample):
        """
        #### Predict the label of a sample if it is a leaf, otherwise predict the sample on the right 
        #### or left node depending on the feature and split threshold
        (recursive function)
        """
        # Check if it is a leaf
        if self.left == None and self.right == None:
            l, c = np.unique(self.labels, return_counts = True)
            
            return l[np.argsort(c)[-1]]

        else:
            # Predict on left or right child node 
            if sample[self.split_feature] <= self.split_threshold:
                return self.left.predict(sample)
            else:
                return self.right.predict(sample)


def gini_impurity(labels):
    """
    #### Compute the Gini impurity of a node based on its labels
    Parameters:
        labels: sample labels of the node

    Return: 
        impurity: Gini impurity of the node
    """
    unique_labels, counts = np.unique(labels, return_counts = True)
    proportions = counts/len(labels)

    impurity = np.sum(proportions * (1-proportions))
    
    return impurity

def best_split_impurity(feature, labels):
    """
    #### Find the split threshold to minimize the Gini impurity if the node is splitted on the given feature: 
        Goes through every possible split of a feature, compute the associated impurity and return the lowest

    Parameters:
        feature: single-feature value for all the samples
        labels: labels of the samples

    Return:
        best_impurity: lowest Gini impurity
        best_split: threshold to obtain the lowest impurity
    """
    # -- Sort the samples
    sorted_ind = np.argsort(feature)

    feature = feature[sorted_ind]
    labels = labels[sorted_ind]
    
    # -- Compute Gini impurity of every potential split
    potential_splits= (feature[1:] + feature[:-1])/2
    impurity = {s:0 for s in potential_splits}
    
    for s in potential_splits:
        mask = feature <= s

        # -- Left node impurity
        left_sub_labels = labels[mask]
        left_impurity = gini_impurity(left_sub_labels)

        # -- Right node impurity
        right_sub_labels = labels[np.logical_not(mask)]
        right_impurity = gini_impurity(right_sub_labels)

        # -- Compute Node Gini impurity
        # weight by the size of the subsamples
        left_weight = len(left_sub_labels)/len(labels)
        right_weight = len(right_sub_labels)/len(labels)
        
        total_impurity = (left_weight*left_impurity)+(right_weight*right_impurity)

        # store the result
        impurity[s] = total_impurity

    # -- Find the lowest Gini impurity
    best_split = min(impurity, key=impurity.get)
    best_impurity = impurity[best_split]

    return best_impurity, best_split