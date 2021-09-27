"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from sklearn.metrics.pairwise import distance_metrics
from utils import euclidean_dist_squared, mode


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        T, _ = X_hat.shape

        distances = np.argsort(euclidean_dist_squared(self.X, X_hat), axis=0)
        y_pred = np.empty(T)

        for t in range(T):
            y = mode(self.y[distances[:self.k, t]])
            y_pred[t] = y

        return y_pred