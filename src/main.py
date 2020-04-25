import yaml

from classifier_handcrafted_features.model import HandcraftedFeatures
from classifier_majority.model import Majority

results = {}
for i, target in enumerate(['Book relevance', 'Type', 'CategoryBroad']):
    results[target] = {}

    results[target]['Majority'] = Majority(target=target).cross_validate()
    results[target]['NB'] = HandcraftedFeatures('NB', target=target).cross_validate()
    results[target]['RF'] = HandcraftedFeatures('RF', target=target).cross_validate()
    results[target]['SVM'] = HandcraftedFeatures('SVM', target=target).cross_validate()
    results[target]['LR'] = HandcraftedFeatures('LR', target=target, standardize=True).cross_validate()

yaml.dump(results, open('../results/results.yaml', 'w+'), default_flow_style=False)
