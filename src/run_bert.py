# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:20:13 2020

@author: Patrik
"""
# This file was used to train the models. 
# Once the models are trained can be used to evaluate their performance

import os
import tensorflow as tf
from classifier_BERT.model import Bert_Model


# model1 = Bert_Model(target = 'Book relevance')
# acc1 = model1.evaluate(epochs = 3)
# print('Accuracy on test set: ', acc1)


# model2 = Bert_Model(target = 'Type', english = False)
# acc2 = model2.evaluate(epochs = 3)
# print('Accuracy on test set: ',acc2)

# model3 = Bert_Model(target = 'CategoryBroad',english = False)
# acc3 = model3.evaluate(epochs = 3)
# print('Accuracy on test set: ',acc3)


# model6 = Bert_Model(target = 'Book relevance', english = True)
# acc6 = model6.evaluate(epochs = 3)
# print('Accuracy on test set: ', acc6)


# model4 = Bert_Model(target = 'Type', english = True)
# acc4 = model4.evaluate(epochs = 3)
# print('Accuracy on test set: ',acc4)

model5 = Bert_Model(target = 'CategoryBroad',english = True)
acc5 = model5.evaluate(epochs = 3)
print('Accuracy on test set: ',acc5)

