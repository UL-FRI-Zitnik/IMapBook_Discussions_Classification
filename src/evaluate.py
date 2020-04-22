"""
File for evaluating models.
"""


def evaluate(data, model):
    xtrain, ytrain, xtest, ytest = data
    model.fit(xtrain, ytrain)
    y_predicted = model.predict(xtest)
    acc(y_predicted, ytest)


def acc(y1, y2):
    pass
