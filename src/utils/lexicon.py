"""
Script takes lexicon from https://www.clarin.si/repository/xmlui/handle/11356/1230
and extracts words from it and saves them to ./data/lexicon/words
and ./data/lexicon/words_pickled.
"""
import pickle

# only words
words = set()
with open('../../data/lexicon/Morphological lexicon Sloleks 2.0/Sloleks2.0.MTE/sloleks_clarin_2.0-sl.tbl') as file:
    for line in file.readlines():
        word = line.split('\t')[0].lower()
        words.add(word)

with open('../../data/lexicon/words_pickled', 'wb+') as file:
    pickle.dump(words, file)

words = sorted(list(words))
with open('../../data/lexicon/words', 'w+') as file:
    for word in words:
        file.write(word)
        file.write('\n')

# words and probabilities
words = {}
with open('../../data/lexicon/Morphological lexicon Sloleks 2.0/Sloleks2.0.MTE/sloleks_clarin_2.0-sl.tbl') as file:
    for line in file.readlines():
        word = line.split('\t')
        words[word[0].lower()] = float(word[3].strip())

# normalizing
s = 0
for p in words.values():
    s += p
for word in words:
    words[word] /= s

with open('../../data/lexicon/words_and_probabilities_pickled', 'wb+') as file:
    pickle.dump(words, file)

with open('../../data/lexicon/words_and_probabilities', 'w+') as file:
    for word, p in words.items():
        file.write('{}\t{}\n'.format(word, p))
