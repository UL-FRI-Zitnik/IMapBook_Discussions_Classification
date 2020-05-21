from classifier_elmo.elmo_embeddings import get_features
from classifier_handcrafted_features.model import HandcraftedFeatures


class ElmoClassifier(HandcraftedFeatures):
    def __init__(self, model, target='Book relevance', *, standardize=False):
        super().__init__(
            model=model,
            target=target,
            standardize=standardize,
            correct_typos=True,
        )

    def get_features(self, data):
        return get_features(data)

    def __str__(self):
        return 'ELMo, {}'.format(self.model_name)
