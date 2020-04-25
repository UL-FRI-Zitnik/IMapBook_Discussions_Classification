"""
Base class for the models.
"""

import numpy as np
from sklearn import metrics

from utils.data import kfolds
from utils.metrics import log_loss, get_mean_se

class Model:
    def __init__(self, imap_columns, target):
        assert target in ('Book relevance', 'Type', 'Category', 'CategoryBroad')
        self.target = target

        assert len(set(imap_columns).difference({
            'School', 'Cohort', 'Book ID', 'Topic', 'Bookclub', 'User ID', 'Name', 'Message', 'Translation',
            'Message Time', 'Page'
        })) == 0
        self.imap_columns = imap_columns

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
        return 'target={}'.format(self.target)

    def __str__(self):
        return '{}, {}'.format(type(self).__name__, self.params_str())

    def cross_validate(self):
        accuracy = []
        log_losses = []

        for xtrain, ytrain, xtest, ytest in kfolds(self.imap_columns, self.target):
            self.fit(xtrain, ytrain)
            y_predicted_probabilities, classes = self.predict_probabilities(xtest)
            y_predicted = classes[np.argmax(y_predicted_probabilities, axis=1)]

            no_same = np.sum(ytest == y_predicted)
            accuracy += [1] * no_same + [0] * (len(ytest) - no_same)
            log_losses += log_loss(y_predicted_probabilities, ytest, classes)

        acc_mean, acc_se = get_mean_se(accuracy)
        ll_mean, ll_se = get_mean_se(log_losses)

        print(self)
        print('accuracy (+- SE): {:.2f} +- {:.3f}'.format(acc_mean, acc_se))
        print('log loss (+- SE): {:.2f} +- {:.3f}'.format(ll_mean, ll_se))
        print()

        return {
            'acc': float(acc_mean),
            'acc_se': float(acc_se),
            'll': float(ll_mean),
            'll_se': float(ll_se),
        }
