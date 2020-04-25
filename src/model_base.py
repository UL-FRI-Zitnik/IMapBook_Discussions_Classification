"""
Base class for the models.
"""

import numpy as np
from sklearn import metrics

from utils.data import kfolds


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
        acc = []
        f1 =[]

        for xtrain, ytrain, xtest, ytest in kfolds(self.imap_columns, self.target):
            self.fit(xtrain, ytrain)
            y_predicted = self.predict(xtest)

            acc.append(metrics.accuracy_score(ytest, y_predicted))
            
            # if self.target == 'Book relevance':
            #   f1.append(metrics.f1_score(ytest, y_predicted, average= 'binary'))
            # else:
            f1.append(metrics.f1_score(ytest, y_predicted, average = 'weighted'))
        print(self)
        print('Accuracy (+- STD): {:.2f} +- {:.3f}'.format(np.mean(acc), np.std(acc)))
        print('F1 score (+- STD): {:.2f} +- {:.3f}'.format(np.mean(f1), np.std(f1)))
        print()
        
        return (np.mean(acc), np.std(acc),np.mean(f1), np.std(f1))
