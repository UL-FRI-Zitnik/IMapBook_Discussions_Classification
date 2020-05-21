import pickle
from os import path

import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids


def get_features(messages):
    if path.exists('../data/pickled_data/ELMo'):
        cache = pickle.load(open('../data/pickled_data/ELMo', 'rb'))
    else:
        cache = {}

    options_file = "../data/elmo/options.json"
    weight_file = "../data/elmo/slovenian-elmo-weights.hdf5"
    elmo = Elmo(options_file, weight_file, 1, dropout=0)

    X = []

    for i in range(len(messages)):
        topic = messages['Topic'].iloc[i].split('/')[0]
        message = str(messages['Message'].iloc[i])

        key = topic + message
        if key in cache:
            X.append(cache[key])
            continue

        character_ids = batch_to_ids([[topic, message]])
        embedding = elmo(character_ids)['elmo_representations'][0].squeeze().detach().numpy()[1]

        cache[key] = embedding
        X.append(embedding)

    pickle.dump(cache, open('../data/pickled_data/ELMo', 'wb+'))

    return np.array(X)
