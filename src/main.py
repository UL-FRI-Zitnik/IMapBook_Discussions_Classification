import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classifier_handcrafted_features.model import HandcraftedFeatures
from classifier_majority.model import Majority

plt.style.use('ggplot')

tmp1 = np.zeros((5, 3))
tmp2 = np.zeros((5, 3))
err_a = np.zeros((5, 3))
err_f = np.zeros((5, 3))
path = r'../results/'

for i, target in enumerate(['Book relevance', 'Type', 'CategoryBroad']):
    label = path + target
    if os.path.isdir(label):
        file = np.loadtxt(label + '/results.csv')
        tmp1[:, i] = file[:, 0]
        err_a[:, i] = file[:, 1]
        tmp2[:, i] = file[:, 2]
        err_f[:, i] = file[:, 3]
    else:
        tmp1[0, i], err_a[0, i], tmp2[0, i], err_f[0, i] = Majority(target=target).cross_validate()
        tmp1[1, i], err_a[1, i], tmp2[1, i], err_f[1, i] = HandcraftedFeatures('Bayes', target=target).cross_validate()
        tmp1[2, i], err_a[2, i], tmp2[2, i], err_f[2, i] = HandcraftedFeatures('RF', target=target).cross_validate()
        tmp1[3, i], err_a[3, i], tmp2[3, i], err_f[3, i] = HandcraftedFeatures('SVM', target=target).cross_validate()
        tmp1[4, i], err_a[4, i], tmp2[4, i], err_f[4, i] = HandcraftedFeatures('LR', target=target,
                                                                               standardize=True).cross_validate()
        os.mkdir(label)
        np.savetxt(label + '/results.csv', np.vstack((tmp1[:, i], err_a[:, i], tmp2[:, i], err_f[:, i])).T)

indx = ['Majority', 'NB', 'RF', 'SVM', 'LR']
df_acc = pd.DataFrame(tmp1, columns=['Book relevance', 'Type', 'CategoryBroad'], index=indx)
df_f1 = pd.DataFrame(tmp2, columns=['Book relevance', 'Type', 'CategoryBroad'], index=indx)

df_acc.plot.barh(title='Accuracy', xerr=err_a.T, capsize=3)
plt.savefig(path + 'accuracy_plot.pdf', format='pdf')
df_f1.plot.barh(title='F1 score', xerr=err_f.T, capsize=3)
plt.savefig(path + 'f1_plot.pdf', format='pdf')
