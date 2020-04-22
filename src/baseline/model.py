"""
Model definition.
"""

from baseline.feature_extraction import get_features


class Baseline:
    def __init__(self):
        pass

    def fit(self, X, y):
        X = get_features(X)

    def predict(self, X):
        pass
