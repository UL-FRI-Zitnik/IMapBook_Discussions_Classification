import pickle
import string

import yaml
from nltk import tokenize
from tqdm import tqdm

from utils.data import get_data


def correction(word : str):
    if len(word) == 0:
        return None, 0
    uppercase_first = word[0].isupper()
    word = word.lower()

    c, n = candidates(word)
    if c is None:
        return None, 0
    w = max(c, key=lambda x: lexicon[x])
    if uppercase_first:
        w = w[0].upper() + w[1:]
    return w, n


def candidates(word : str):
    words = []

    # expanding
    idx = []
    for i, c in enumerate(word):
        if c in 'scz':
            idx.append(i)
    for i in range(2 ** len(idx)):
        w = list(word)
        for j in range(len(idx)):
            if i % 2 == 1:
                w[idx[j]] = {
                    's': 'š',
                    'c': 'č',
                    'z': 'ž',
                }[w[idx[j]]]
            i //= 2
        words.append(''.join(w))

    w = known(words)
    if w:
        return w, 0

    w = known(join(words, edits1))
    if w:
        return w, 1

    w = known(join(words, edits2))
    if w:
        return w, 2

    return None, 0


def join(words, f):
    r = set()
    for word in words:
        r.update(set(f(word)))
    return r


def known(words):
    return set(w for w in words if w in lexicon)


def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


data = get_data()
data = data.dropna(subset=['Message'])
data = data.reset_index(drop=True)
data = data.loc[:, ['Message']]

lexicon = pickle.load(open('../data/lexicon/words_and_probabilities_pickled', 'rb'))
letters = set(string.ascii_lowercase + 'čšž')
letters_all = letters.union(set((string.ascii_lowercase + 'čšž').upper()))
punctuation = set('!"(),.:?')

corrected = {}

for i in tqdm(range(len(data))):
    message = str(data['Message'].iloc[i])

    correct = []
    removed = []
    no_typos = 0
    tokens = tokenize.casual_tokenize(message)

    for token_ in tokens:
        for token in token_.split('.'):
            if len(token) > 14:
                removed.append(token)
            elif len(set(token).difference(letters_all)) > 0:
                if len(token) == 1 and token in punctuation:
                    correct.append(token)
                else:
                    removed.append(token)
            else:
                c, no_mistakes = correction(token)
                if c is not None and lexicon[c.lower()] > 1e-7:
                    correct.append(c)
                    no_typos += no_mistakes
                else:
                    removed.append(token)

    corrected[message] = {
        'corrected': correct,
        'removed words': removed,
        'no typos': no_typos
    }

yaml.dump(corrected, open('../data/messages_without_typos.yaml', 'w+'), default_flow_style=False)
