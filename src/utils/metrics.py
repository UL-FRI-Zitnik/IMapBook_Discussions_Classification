import numpy as np


def log_loss(p, y, classes):
    classes = list(classes)
    ret = []
    for i in range(len(y)):
        ret.append(-np.log(p[i, classes.index(y.iloc[i])]))
    return ret


def get_mean_se(scores):
    scores = np.array(scores)
    mean = np.mean(scores)
    se = np.std(scores) / np.sqrt(len(scores))
    return mean, se
