from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from classifier_handcrafted_features.feature_extraction import get_features
from utils.model_base import Model


class HandcraftedFeatures(Model):
    def __init__(self, model, target='Book relevance', *, standardize=False):
        super().__init__(
            imap_columns=('Message', 'Topic'),
            target=target
        )

        self.standardize = standardize

        if model == 'NB':
            self.model = MultinomialNB()
        elif model == 'RF':
            self.model = RandomForestClassifier(n_estimators=150,
                                                min_samples_leaf=3,
                                                min_samples_split=10,
                                                random_state=0,
                                                n_jobs=-1)
        elif model == 'SVM':
            self.model = svm.SVC(C=5, gamma='auto')
        elif model == 'LR':
            self.model = LogisticRegression(max_iter=1000, n_jobs=-1)
        else:
            raise

    def fit(self, messages, y):
        X = get_features(messages)

        if self.standardize:
            self.normalize(X, init=True)

        self.model.fit(X, y)
        # print(self.model.score(X, y))

    def predict(self, messages):
        X = get_features(messages)

        if self.standardize:
            self.normalize(X)

        return self.model.predict(X)

    def predict_probabilities(self, messages):
        X = get_features(messages)

        if self.standardize:
            self.normalize(X)

        return self.model.predict_proba(X), self.model.classes_

    def params_str(self):
        return 'target={}, model={}'.format(self.target, type(self.model).__name__)
