import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

assert os.getcwd().split('/')[-1] == 'src', "Run from 'src' folder."

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
    plt.savefig(os.path.join('../results', score_nice + '_plot.pdf'), format='pdf', bbox_inches='tight')
