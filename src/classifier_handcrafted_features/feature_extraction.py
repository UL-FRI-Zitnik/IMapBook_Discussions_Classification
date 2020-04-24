import pickle
import re
from os import path

import pandas as pd
from nltk import tokenize
from nltk.metrics.distance import edit_distance as Lev


def levenshtein_dist(message1, message2, stop_words=None, thr=0.5, substitution_cost=2):
    """
    Counts the number of words in the message, whose levenshtein_dist distance from any word in the "question" is smaller than threshold
      - tokens1 and tokens2 are tokenized messages/questions.. order should not matter
      - Threshold (thr) is relative to the word length (default 1/2 the length)
      - SW is an array of stop-words: if given the words present in this array are ignored
      - sub_cost is the cost for substitution in the calculation of levenshtein distance
    """

    tokens1 = tokenize.casual_tokenize(message1)
    tokens2 = tokenize.casual_tokenize(message2)

    dist = 0
    for t1 in tokens1:
        if stop_words is not None and t1 in stop_words:
            continue

        for t2 in tokens2:
            if stop_words is not None and t2 in stop_words:
                continue

            n = max(len(t2), len(t1))
            l = Lev(t2, t1, substitution_cost=substitution_cost)
            if l <= (n * thr):
                dist += 1

    return dist


def prop_name_count(message, names):
    """
    Function counts the appearance of proper names(names is array of proper names) in tokenized messages (tokens)
    """
    tokens = tokenize.casual_tokenize(message)
    d = 0
    for token in tokens:
        if token.capitalize() in names:
            d += 1
    return d


def get_features(messages):
    if path.exists('../data/pickled_data/HandcraftedFeatures'):
        cache = pickle.load(open('../data/pickled_data/HandcraftedFeatures', 'rb'))
    else:
        cache = {}

    X = []
    lexicon = pickle.load(open('../data/lexicon/words_pickled', 'rb'))
    stop_words = pd.read_csv('../data/sloStopWords.txt', sep='\n').to_numpy()
    names = pd.read_csv('../data/Slovenska_Imena.txt', sep=',', header=None).to_numpy()

    for i in range(len(messages)):
        topic = messages['Topic'].iloc[i].split('/')[0]
        message = str(messages['Message'].iloc[i])

        key = topic + message
        if key in cache:
            X.append(cache[key])
            continue

        x = []

        # #tokens
        tokens = tokenize.casual_tokenize(message)
        x.append(len(tokens))

        # #mistakes in words
        mistakes = 0
        for word in tokens:
            if word.lower() not in lexicon:
                mistakes += 1
        x.append(mistakes)

        # max len of a word
        x.append(max(map(len, tokens + [''])))

        # #chars
        x.append(len(message))

        # #?
        x.append(message.count('?'))

        # #!
        x.append(message.count('!'))

        # #,
        x.append(message.count(','))

        # #.
        x.append(message.count('.'))

        # #caps
        x.append(len(re.findall('[A-Z]', message)))

        # #interior caps
        x.append(len(re.findall('\w[A-Z]', message)))

        # #strange letters
        x.append(len(re.findall('[^\w .,?]', message)))

        # #interior numbers
        x.append(len(re.findall('[a-zA-Z][0-9]', message)))

        # lev. distance
        x.append(levenshtein_dist(message, topic, stop_words=stop_words))

        # #names
        x.append(prop_name_count(message, names))

        # #question words (without 'kdo')
        n = 0
        for w in tokens:
            if w in {'kaj', 'koga', 'česa', 'komu', 'čemu', 'kom', 'čim',
                     'zakaj', 'kam', 'kje', 'kod', 'kako', 'kdaj'}:
                n += 1
        x.append(n)

        # #'kdo'
        n = 0
        for w in tokens:
            if w == 'kdo':
                n += 1
        x.append(n)

        cache[key] = x
        X.append(x)

    pickle.dump(cache, open('../data/pickled_data/HandcraftedFeatures', 'wb+'))

    return pd.DataFrame(
        X,
        # columns=[
        # '#words',
        # '#mistakes in words',
        # 'max len of a word',
        # '#chars',
        # '#?',
        # '#!',
        # '#,',
        # '#caps',
        # '#interior caps',
        # '#strange letters',
        # '#interior numbers',
        # 'lev. distance',
        # '#names',
        # ]
    )
