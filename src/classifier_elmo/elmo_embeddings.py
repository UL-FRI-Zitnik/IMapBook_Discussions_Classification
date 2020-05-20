from allennlp.modules.elmo import Elmo, batch_to_ids


def get_features(messages):
    options_file = "../data/elmo/options.json"
    weight_file = "../data/elmo/slovenian-elmo-weights.hdf5"
    elmo = Elmo(options_file, weight_file, 1, dropout=0)

    messages = [[str(messages['Message'].iloc[i])] for i in range(len(messages))]

    character_ids = batch_to_ids(messages)

    embeddings = elmo(character_ids)
    embeddings = embeddings['elmo_representations'][0].squeeze().detach().numpy()

    return embeddings
