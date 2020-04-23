# utils for feature extraction:

import numpy as np
import pandas as pd

# loading slovene stopwords:
SW = pd.read_csv('../../data/sloStopWords.txt', sep='\n').to_numpy()

# example of what is a book (dataframe with 3 columns: Topic, Message, Book Relevance(label))
# data = pd.read_excel('../../data/imapBook/AllDiscussionDataCODED_USE_THIS_14Feb2020_MH.xls')
# data = data.dropna(axis= 1, how = 'all')
# data = data.dropna(axis= 0, how = 'all')

# grouping the messages by topic
# g_data = data.loc[data['Book ID'].isin([8.0, 12.0, 15.0])].groupby('Book ID')
# g_messages = g_data[['Topic', 'Message', 'Book relevance']]
# book1 = g_messages.get_group(8.0)
# book2 =  g_messages.get_group(12.0)
# book3 =  g_messages.get_group(15.0)

# Very simple tokenizer, that tokenizes messages:
from nltk import tokenize


def tokenize_data(book):
    question_tokens = []
    text_tokens = []
    for ind, message in enumerate(book["Message"]):
        text_tokens.append(tokenize.casual_tokenize(str(message)))
        question_tokens.append(tokenize.casual_tokenize(str(book['Topic'].iloc[ind]).split('/')[0]))
    return question_tokens, text_tokens


from nltk.metrics.distance import edit_distance as Lev


def levenshtein_dist(tokens1, tokens2, SW=None, thr=0.5, sub_cost=2):
    # Features levenshtein_dist:
    #   Counts the number of words in the message, whose levenshtein_dist distance from any word in the "question" is smaller than threshold
    #   - tokens1 and tokens2 are tokenized messages/questions.. order should not matter
    #   - Threshold (thr) is relative to the word length (default 1/2 the length)
    #   - SW is an array of stop-words: if given the words present in this array are ignored
    #   - sub_cost is the cost for substitution in the calculation of levenshtein distance

    d = np.zeros(len(tokens1))
    for i, t1 in enumerate(tokens1):
        tmp_dist = 0
        #         n = 0
        for j in t1:
            if SW is not None:
                if j in SW:
                    continue
            for k in tokens2[i]:
                if SW is not None:
                    if k in SW:
                        continue
                n = max(len(k), len(j))
                l = Lev(k, j, substitution_cost=sub_cost)
                if l <= (n * thr):
                    tmp_dist += 1
        d[i] = tmp_dist
    return d


def prop_name_count(tokens, names):
    #   Function counts the appearance of proper names(names is array of proper names) in tokenized messages (tokens)
    d = np.zeros(len(tokens))
    for i, M in enumerate(tokens):
        for j in M:
            if j.capitalize() in names:
                d[i] += 1
    return d

# example how to use it:
# nms = pd.read_csv('../utils/Slovenska_Imena.txt', sep = ',', header= None).to_numpy()
# prop_name_count(text_tokens, nms)
