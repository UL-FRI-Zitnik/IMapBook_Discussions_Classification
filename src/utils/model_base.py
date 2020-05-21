"""
Base class for the models.
"""

import numpy as np
from tqdm import tqdm

from utils.data import kfolds
from utils.metrics import get_mean_se


class Model:
    def __init__(self, imap_columns, target, *, correct_typos=False):
        assert target in ('Book relevance', 'Type', 'Category', 'CategoryBroad')
        self.target = target
        self.correct_typos = correct_typos

        assert len(set(imap_columns).difference({
            'School', 'Cohort', 'Book ID', 'Topic', 'Bookclub', 'User ID', 'Name', 'Message', 'Translation',
            'Message Time', 'Page'
        })) == 0
        self.imap_columns = imap_columns

        self.model = None
        self.mean = self.std = None

    def fit(self, messages, y):
        raise

    def predict(self, messages):
        raise

    def predict_probabilities(self, messages):
        raise

    def normalize(self, X, *, init=False):
        if init:
            self.mean = np.mean(X, axis=0)
        X -= self.mean

        if init:
            self.std = np.std(X, axis=0)
        X /= self.std

    def params_str(self):
        return ''

    def __str__(self):
        return '{}{}'.format(type(self).__name__, self.params_str())

    def cross_validate(self):
        print(self)
        acc = []

        for xtrain, ytrain, xtest, ytest in kfolds(self.imap_columns, self.target, correct_typos=self.correct_typos):
            self.fit(xtrain, ytrain)
            y_predicted = self.predict(xtest)

            no_same = np.sum(ytest == y_predicted)
            acc += [1] * no_same + [0] * (len(ytest) - no_same)  # todo: ...

        acc_mean, acc_se = get_mean_se(acc)

        print('accuracy (+- SE): {:.2f} +- {:.3f}'.format(acc_mean, acc_se))
        print()

        return {
            'acc': float(acc_mean),
            'acc_se': float(acc_se),
        }

    def feature_importances(self):
        importances = []

        for xtrain, ytrain, xtest, ytest in kfolds(self.imap_columns, self.target):
            self.fit(xtrain, ytrain)
            importances.append(self.model.feature_importances_)

        return np.mean(importances, 0)
