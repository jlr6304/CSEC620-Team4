
import numpy as np
from math import inf

class Node:
    def __init__(self, data, labels, parent_imp = inf):
        self.left = None
        self.right = None
        self.data = data
        self.labels = labels
        self.parentImpurity = parent_imp

    def split(self, feature_subcount):
        if not self.isLeaf():
            
            n_samples = self.data.shape[0]
            n_features = self.data.shape[1]

            # Choose randomly a subset of the features
            features = np.unique(np.random.choice(n_features, feature_subcount, replace=True))

            # Compute the best threshold (Impurity) for each feature
            impurity = {f: 0 for f in features}
            location = {f: 0 for f in features}
            for f in features:
                impurity[f], location[f] = best_split_impurity(self.data[:, f], self.labels)
            
            best_f = min(impurity, key=impurity.get)
            best_l = location[best_f]
            
            # Split on the feature with the lowest Impurity
            target_samples = self.data[:, best_f]

            mask = target_samples <= best_l
            
            # self.condition() updated with the new threshold 
            left_sub_samples = self.data[mask, :]
            left_sub_labels = self.labels[mask]

            right_sub_samples = self.data[np.logical_not(mask), :]
            right_sub_labels = self.labels[np.logical_not(mask)]
            
            # Create child nodes
            self.left = Node(left_sub_samples, left_sub_labels)
            self.right = Node(right_sub_samples, right_sub_labels)
           
            # Split child nodes
            self.left.split()
            self.right.split()

    def isLeaf(self):
        """
        Test if it is a leaf
        """
        #check if the branch has reached the max_depth

        #check if the number of samples in the group to split is less than the min_node

        #check if the optimal split results in a group with no samples, in which case use the parent node
        
        #check if the optimal split has a worse Gini impurity than the parent node, in which case use the parent node.

def fit(train_set, train_labels, feature_subcount, max_depth=10, min_node=10):
    tree = Node(train_set, train_labels)

    tree.split(feature_subcount)

    return tree

def predict():
    pass


def gini_impurity():
    pass

def best_split_impurity():
    pass