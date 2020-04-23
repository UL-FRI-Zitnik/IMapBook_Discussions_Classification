from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from classifier_handcrafted_features.feature_extraction import get_features
from model_base import Model


class HandcraftedFeatures(Model):
    def __init__(self, model):
        super().__init__(
            imap_columns=('Message', 'Topic'),
            target='Book relevance'
        )

        if model == 'Bayes':
            self.model = MultinomialNB()
        elif model == 'RF':
            self.model = RandomForestClassifier(random_state=0)
        elif model == 'SVM':
            self.model = svm.SVC()
        else:
            raise

    def fit(self, messages, y):
        X = get_features(messages)
        self.model.fit(X, y)

    def predict(self, messages):
        X = get_features(messages)
        return self.model.predict(X)

    def params_str(self):
        return 'target={}, model={}'.format(self.target, type(self.model).__name__)
