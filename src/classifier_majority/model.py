from collections import defaultdict

import numpy as np

from utils.model_base import Model


class Majority(Model):
    def __init__(self, target='Book relevance'):
        super().__init__(
            imap_columns=['Message'],
            target=target
        )

        self.majority_class = None
        self.distribution = None

    def fit(self, _, y):
        counts = defaultdict(int)
        for item in y:
            counts[item] += 1

        self.classes = np.array(list(counts.keys()))
        self.probabilities = [counts[key] / len(y) for key in self.classes]

        counts = [(key, value) for key, value in counts.items()]
        counts.sort(key=lambda x: -x[1])
        self.majority_class = counts[0][0]

    def predict(self, messages):
        return np.array([self.majority_class] * len(messages))

    def predict_probabilities(self, messages):
        return np.array([self.probabilities] * len(messages)), self.classes
