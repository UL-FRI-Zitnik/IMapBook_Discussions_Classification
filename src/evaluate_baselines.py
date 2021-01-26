import os

import yaml

from classifier_handcrafted_features.model import HandcraftedFeatures
from classifier_majority.model import Majority
from plots.plot_baselines import plot

if os.path.exists('../results/results_baselines.yaml'):
    results = yaml.load(open('../results/results_baselines.yaml'), yaml.Loader)
else:
    results = {}

for i, target in enumerate(['Category', 'Book relevance', 'Type', 'CategoryBroad']):
    print('TARGET:', target, '\n')

    if target not in results:
        results[target] = {}

    models = [
        Majority(target=target),
        HandcraftedFeatures('NB', target=target),
        HandcraftedFeatures('RF', target=target),
        HandcraftedFeatures('SVM', target=target),
        HandcraftedFeatures('LR', target=target, standardize=True),
    ]

    for model in models:
        results[target][str(model)] = model.cross_validate()

yaml.dump(results, open('../results/results_baselines.yaml', 'w+'), default_flow_style=False)

plot()
