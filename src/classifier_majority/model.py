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

    def fit(self, _, y):
        counts = defaultdict(int)
        for item in y:
            counts[item] += 1
        counts = [(key, value) for key, value in counts.items()]
        counts.sort(key=lambda x: -x[1])
        self.majority_class = counts[0][0]

    def predict(self, messages):
        return np.array([self.majority_class] * len(messages))
