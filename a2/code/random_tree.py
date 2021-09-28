from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpInfoGain)

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    def __init__(self, max_depth, num_trees) -> None:
        self.max_depth = max_depth
        self.num_trees = num_trees 
        self.trees = []

    def fit(self, X, y): 
        for _ in range(self.num_trees):
            tree = RandomTree(self.max_depth)
            tree.fit(X,y)
            self.trees.append(tree)
    
    def predict(self, X):
        N, _ = X.shape

        tree_predictions = np.zeros((N, self.num_trees))
        
        for i in range(self.num_trees):
            tree_predictions[:, i] = self.trees[i].predict(X)

        return [utils.mode(tree_predictions[i,:]) for i in range(N)]