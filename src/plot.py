import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

results = yaml.load(open('../results/results.yaml'), yaml.Loader)

models = list(results[list(results.keys())[0]].keys())
targets = list(results.keys())

plt.style.use('ggplot')

for score in ['acc']:
    score_nice = {
        'acc': 'Accuracy',
    }[score]

    scores = pd.DataFrame([
        [results[target][model][score] for target in targets]
        for model in models
    ], columns=targets, index=models)
    scores = scores.sort_values('Book relevance', ascending=False)
    scores = scores.sort_values('Majority', axis='columns')

    errors = np.array([
        [results[target][model][score + '_se'] for target in targets]
        for model in models
    ]).T

    scores.plot.barh(title=score_nice, xerr=errors, capsize=3)
    plt.savefig(os.path.join('../results', score_nice + '_plot.pdf'), format='pdf')

 # Plotting feature importance for RF
# from classifier_handcrafted_features.model import HandcraftedFeatures

# inds = ['Book relevance', 'Type', 'CategoryBroad']
# cols =['#words','#mistakes in words', 'max len of a word', '#chars', '#?', '#!', '#,','#.','#caps','#interior caps','#strange letters', '#interior numbers','lev. distance','#names','#quest_w','#who']

# imp = []
# for i, target in enumerate(inds):
#   imp.append( HandcraftedFeatures('RF', target=target).cross_validate(True))
   
# importance = pd.DataFrame(imp, columns = cols, index = inds).T
# importance = importance.sort_values('Book relevance', ascending = False)
# importance.plot.barh(title = 'Features Importance for RF Model', fontsize = 14)
# plt.tight_layout()
# plt.savefig(os.path.join('../results', 'feature_imp_RF_plot.pdf'), format='pdf')
