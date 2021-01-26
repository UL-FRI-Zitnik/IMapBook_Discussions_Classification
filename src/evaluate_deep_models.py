# Out of bag evaluation

import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from classifier_BERT.model import Bert_Model
from classifier_elmo.model import ElmoClassifier
from classifier_handcrafted_features.model import HandcraftedFeatures
from plots.plot_deep_models import plot
from utils.data import select_columns

from sklearn.metrics import confusion_matrix

def print_cnf(label_encoder, test_y, pred_y):
    classes = label_encoder.classes_
    cnf = confusion_matrix(label_encoder.inverse_transform(test_y),
                           label_encoder.inverse_transform(pred_y),
                           labels=classes)
    print(classes)
    print(cnf)

plt.style.use('ggplot')

targets = ['Category', 'Book relevance', 'Type', 'CategoryBroad']

results = []
for t in targets:
    print('Evaluating on target:', t)

    train_X, train_y, test_X, test_y = select_columns(x=['Message', 'Translation', 'Topic'], y=t)
    enc = LabelEncoder()
    enc.fit(train_y.values)
    test_y = enc.transform(test_y.values)

    print('Handcrafted Features')
    rf = HandcraftedFeatures('RF', target=t)
    rf.fit(train_X[['Message', 'Topic']], train_y)
    pred_rf = enc.transform(rf.predict(test_X[['Message', 'Topic']]))
    f1_rf = f1_score(test_y, pred_rf, average='weighted')
    print_cnf(enc, test_y, pred_rf)
    print('F1:', f1_rf)

    print('ELMo')
    elmo = ElmoClassifier('RF', target=t)
    elmo.fit(train_X[['Message', 'Topic']], train_y)
    pred_elmo = enc.transform(elmo.predict(test_X[['Message', 'Topic']]))
    f1_elmo = f1_score(test_y, pred_elmo, average='weighted')
    print_cnf(enc, test_y, pred_elmo)
    print('F1:', f1_elmo)

    print('BERT on Slovene messages')
    bert_slo = Bert_Model(t)
    bert_slo.fit(train_X, train_y)
    pred_bert_slo = bert_slo.predict(test_X)
    f1_bert_slo = f1_score(test_y, pred_bert_slo, average='weighted')
    print_cnf(enc, test_y, pred_bert_slo)
    print('F1:', f1_bert_slo)

    print('BERT on English messages')
    bert_eng = Bert_Model(t, True)
    bert_eng.fit(train_X, train_y)
    pred_bert_eng = bert_eng.predict(test_X)
    f1_bert_eng = f1_score(test_y, pred_bert_eng, average='weighted')
    print_cnf(enc, test_y, pred_bert_eng)
    print('F1:', f1_bert_eng)

    results.append([f1_rf,
                    f1_elmo,
                    f1_bert_slo,
                    f1_bert_eng])

results = pd.DataFrame(results, columns=['RF', 'ELMo', 'BERT_slo', 'BERT_eng'])
results.index = targets

pickle.dump(results, open('../results/results_deep_models', 'wb+'))

plot()
