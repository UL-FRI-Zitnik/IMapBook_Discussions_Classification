from classifier_handcrafted_features.model import HandcraftedFeatures
from classifier_majority.model import Majority

Majority().cross_validate()
HandcraftedFeatures('Bayes').cross_validate()
HandcraftedFeatures('RF').cross_validate()
HandcraftedFeatures('SVM').cross_validate()
HandcraftedFeatures('LR', standardize=True).cross_validate()

Majority(target='Type').cross_validate()
HandcraftedFeatures('Bayes', target='Type').cross_validate()
HandcraftedFeatures('RF', target='Type').cross_validate()
HandcraftedFeatures('SVM', target='Type').cross_validate()
HandcraftedFeatures('LR', target='Type', standardize=True).cross_validate()

Majority(target='CategoryBroad').cross_validate()
HandcraftedFeatures('Bayes', target='CategoryBroad').cross_validate()
HandcraftedFeatures('RF', target='CategoryBroad').cross_validate()
HandcraftedFeatures('SVM', target='CategoryBroad').cross_validate()
HandcraftedFeatures('LR', target='CategoryBroad', standardize=True).cross_validate()
