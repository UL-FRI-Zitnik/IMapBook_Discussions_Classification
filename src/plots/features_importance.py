import os

import matplotlib.pyplot as plt
import pandas as pd

if os.getcwd().split('/')[-1] == 'plots':
    os.chdir('..')

from classifier_handcrafted_features.model import HandcraftedFeatures

inds = ['Book relevance', 'Type', 'CategoryBroad']
cols = ['#tokens', '#mistakes in words', 'max len of a word', '#chars', '#?', '#!', '#,', '#.', '#caps',
        '#interior caps', '#strange letters', '#interior numbers', 'lev. distance', '#names', '#quest_w', '#who']

imp = []
for i, target in enumerate(inds):
    imp.append(HandcraftedFeatures('RF', target=target).feature_importances())

importance = pd.DataFrame(imp, columns=cols, index=inds).T
importance = importance.sort_values('Book relevance', ascending=False)
importance.plot.barh(title='Features Importance for RF Model', fontsize=14)

plt.style.use('ggplot')
plt.tight_layout()
plt.savefig(os.path.join('../results', 'features_imp_RF_plot.pdf'), format='pdf')
