"""
File for functions for retrieving data.
"""

import pandas as pd


def get_data():
    return pd.read_excel('../data/imapBook/AllDiscussionDataCODED_USE_THIS_14Feb2020_MH.xls')


def split_train_test(x, y, split=0.8):
    split = int(len(x) * split)
    xtrain = x[:split]
    ytrain = y[:split]
    xtest = x[split:]
    ytest = y[split:]
    return xtrain, ytrain, xtest, ytest


def select_columns(x=('Message',), y=('Book relevance',)):
    data = get_data()
    x = data.loc[:, x]
    y = data.loc[:, y]
    return split_train_test(x, y)


if __name__ == '__main__':
    # example:
    xtrain, ytrain, xtest, ytest = select_columns()
