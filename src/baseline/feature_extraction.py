import pickle
import re

from nltk.corpus import stopwords


def get_features(sentences):
    X = []
    stop = stopwords.words('slovene')
    lexicon = pickle.load(open('../data/lexicon/words_pickled', 'rb'))

    for sentence in sentences.loc[:, 'Message']:
        x = []

        # no. words
        words = re.findall('\w+', sentence)
        x.append(len(words))

        # no. mistakes in words
        mistakes = 0
        for word in words:
            if word.lower() not in lexicon:
                mistakes += 1
        x.append(mistakes)

        # max len of word
        x.append(max(map(len, words)))

        # len
        x.append(len(sentence))

        # no. ?
        x.append(sentence.count('?'))

        # no. !
        x.append(sentence.count('!'))

        # no. ,
        x.append(sentence.count(','))

        # no. capitals
        x.append(len(re.findall('[A-Z]', sentence)))

        # no. capitals that are not at the beginning of a word
        x.append(len(re.findall('\w[A-Z]', sentence)))

        # no. strange letters
        x.append(len(re.findall('[^\w .,?]', sentence)))

        # no. numbers after letter
        x.append(len(re.findall('[a-zA-Z][0-9]', sentence)))

        X.append(x)
    return X
