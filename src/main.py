from classifier_handcrafted_features.model import HandcraftedFeatures
from classifier_majority.model import Majority

Majority().cross_validate()
HandcraftedFeatures('Bayes').cross_validate()
HandcraftedFeatures('RF').cross_validate()
HandcraftedFeatures('SVM').cross_validate()
