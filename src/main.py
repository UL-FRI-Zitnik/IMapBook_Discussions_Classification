from classifier_handcrafted_features.model import HandcraftedFeatures
from classifier_majority.model import Majority

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


target = ['Book relevance', 'Type', 'CategoryBroad']
tmp1 = np.zeros((5,3))
tmp2 = np.zeros((5,3))
err_a = np.zeros((5,3))
err_f = np.zeros((5,3))
path =  r'../Results/'

if not os.path.isdir(path):
  os.mkdir(path)

for t, tt in enumerate(target):
  label = path + tt 
  if os.path.isdir(label):
    file = np.loadtxt(label+'/results.csv')
    tmp1[:,t] = file[:,0]
    err_a[:,t] = file[:,1]
    tmp2[:,t] = file[:,2]
    err_f[:,t] = file[:,3]
  else:
    tmp1[0,t], err_a[0, t], tmp2[0,t], err_f[0,t ] = Majority(target= tt).cross_validate()
    tmp1[1,t], err_a[1, t], tmp2[1,t], err_f[1,t ] = HandcraftedFeatures('Bayes', target= tt).cross_validate()
    tmp1[2,t], err_a[2, t], tmp2[2,t], err_f[2,t ] = HandcraftedFeatures('RF', target= tt).cross_validate()
    tmp1[3,t], err_a[3, t], tmp2[3,t], err_f[3,t ] = HandcraftedFeatures('SVM', target= tt).cross_validate()
    tmp1[4,t], err_a[4, t], tmp2[4,t], err_f[4,t ] = HandcraftedFeatures('LR',target= tt, standardize=True).cross_validate() 
    os.mkdir(label)
    np.savetxt(label+'/results.csv', np.vstack((tmp1[:,t],err_a[:,t],tmp2[:,t],err_f[:,t])).T)

indx = ['Majority', 'NB', 'RF', 'SVM', 'LR']
df_acc = pd.DataFrame(tmp1, columns = ['Book relevance', 'Type', 'CategoryBroad'], index = indx)
df_f1 = pd.DataFrame(tmp2, columns = ['Book relevance', 'Type', 'CategoryBroad'], index = indx)


df_acc.plot.barh(title = 'Accuracy', xerr = err_a.T, capsize = 3)
plt.savefig(path+'accuracy_plot.pdf', format = 'pdf')
df_f1.plot.barh(title = 'F1 score', xerr = err_f.T, capsize = 3)
plt.savefig(path+'f1_plot.pdf', format = 'pdf')