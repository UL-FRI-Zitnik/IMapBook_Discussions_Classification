import pickle
import re

import numpy as np
import pandas as pd
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.metrics.distance import edit_distance as Lev


def levenshtein_dist(sentence1, sentence2, stop_words=None, thr=0.5, substitution_cost=2):
    """
    Counts the number of words in the message, whose levenshtein_dist distance from any word in the "question" is smaller than threshold
      - tokens1 and tokens2 are tokenized messages/questions.. order should not matter
      - Threshold (thr) is relative to the word length (default 1/2 the length)
      - SW is an array of stop-words: if given the words present in this array are ignored
      - sub_cost is the cost for substitution in the calculation of levenshtein distance
    """

    tokens1 = tokenize.casual_tokenize(sentence1)
    tokens2 = tokenize.casual_tokenize(sentence2)

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


def get_features(sentences):
    X = []
    stop = stopwords.words('slovene')
    lexicon = pickle.load(open('../data/lexicon/words_pickled', 'rb'))
    stop_words = pd.read_csv('../data/sloStopWords.txt', sep='\n').to_numpy()
    names = pd.read_csv('../data/Slovenska_Imena.txt', sep=',', header=None).to_numpy()

    for i in range(len(sentences)):
        message = sentences.loc[i, 'Message']
        topic = sentences.loc[i, 'Topic'].split('/')[0]

        message = str(message)  # todo: this is workaround for NA interpreted as missing value

        x = []

        # no. words
        words = re.findall('\w+', message)
        x.append(len(words))

        # no. mistakes in words
        mistakes = 0
        for word in words:
            if word.lower() not in lexicon:
                mistakes += 1
        x.append(mistakes)

        # max len of word
        x.append(max(map(len, words + [''])))

        # len
        x.append(len(message))

        # no. ?
        x.append(message.count('?'))

        # no. !
        x.append(message.count('!'))

        # no. ,
        x.append(message.count(','))

        # no. capitals
        x.append(len(re.findall('[A-Z]', message)))

        # no. capitals that are not at the beginning of a word
        x.append(len(re.findall('\w[A-Z]', message)))

        # no. strange letters
        x.append(len(re.findall('[^\w .,?]', message)))

        # no. numbers after letter
        x.append(len(re.findall('[a-zA-Z][0-9]', message)))

        # levenshtein distance
        x.append(levenshtein_dist(message, topic, stop_words=stop_words))

        # name count
        x.append(prop_name_count(message, names))

        X.append(x)

    return np.array(X, dtype=float)
