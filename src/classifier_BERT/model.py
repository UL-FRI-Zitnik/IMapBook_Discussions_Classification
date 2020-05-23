# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:39:50 2020

@author: Patrik
"""
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from transformers import TFBertForSequenceClassification as bert_class, BertTokenizer
from nltk import tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm

from utils.model_base import Model
from utils.data import *


class Bert_Model(Model):
  
    def __init__(self, target,english = False):
      #possible opt: adam, SGD, RMSprop
        super().__init__(
          imap_columns = ['Message'],
          target=target
          )
        
        self.english = english
        if self.english:
            self.imap_columns = ['Translation']
            self.path = r'classifier_BERT/pretrained_models/' + self.target + '_english'
        else:
            self.path = r'classifier_BERT/pretrained_models/' + self.target
            
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        
        if self.target == 'Type':
            self.n_class = 3
        elif self.target == 'CategoryBroad':
            self.n_class = 6
        else:
            self.n_class = 2
        
        # self.model = customBert(self.n_class)
        
        self.model = bert_class.from_pretrained('classifier_BERT/pretrained_models/slo-hr-en-bert-pytorch', num_labels = self.n_class, from_pt = True)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate= 1e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        if self.model.num_labels > 2:
            self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=["accuracy"])
        else:
            self.model.compile(loss=loss,
                       optimizer=optimizer,
                       metrics=["accuracy"])
        
        self.Lab_Encoder = LabelEncoder()
        
      
      
    def convert_to_input(self, data, pad_token=0, pad_token_segment_id=0, max_length=128):
      input_ids,attention_masks,token_type_ids=[],[],[]
      
      for x in tqdm(data,position=0, leave=True):
        inputs = self.tokenizer.encode_plus(x,add_special_tokens=True, max_length=max_length)
        
        i, t = inputs["input_ids"], inputs["token_type_ids"]
        m = [1] * len(i)
    
        padding_length = max_length - len(i)
    
        i = i + ([pad_token] * padding_length)
        m = m + ([0] * padding_length)
        t = t + ([pad_token_segment_id] * padding_length)
        
        input_ids.append(i)
        attention_masks.append(m)
        token_type_ids.append(t)
      
      return [np.asarray(input_ids), 
                np.asarray(attention_masks), 
                np.asarray(token_type_ids)]
    
    def example_to_features(self, input_ids,attention_masks,token_type_ids,y):
      return {"input_ids": input_ids,
              "attention_mask": attention_masks,
              "token_type_ids": token_type_ids},y
    
    def fit(self, messages, y, epochs = 2, validation_percent = 0.15, allow_import = True):
        if self.english:
            messages = (messages['Translation'].values)
        else:
            messages = (messages['Message'].values)
            
        y = self.Lab_Encoder.fit_transform(y.to_numpy().astype(str)).astype(float)
              
        # preparing the data:
        # split in train and validation
        
        if os.path.exists(self.path) and allow_import:
            print("Loading Pretrained Model")
            self.model = bert_class.from_pretrained(self.path)
            return(self)
        else:
            
            X_train, X_val, y_train, y_val = train_test_split(messages, y, test_size = validation_percent, random_state = 0)
            
            #tokenizing the data and making tf.dataset
            X_train_input= self.convert_to_input(X_train.astype(str))
            X_val_input= self.convert_to_input(X_val.astype(str))
            
            train_ds = tf.data.Dataset.from_tensor_slices((X_train_input[0],X_train_input[1],X_train_input[2],y_train)).map(self.example_to_features).shuffle(100).batch(12).repeat(5)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val_input[0],X_val_input[1],X_val_input[2],y_val)).map(self.example_to_features).batch(12)
            
            self.model.fit(train_ds, epochs = epochs, validation_data = val_ds, verbose = 1) 
            os.mkdir(self.path)
            self.model.save_pretrained(self.path)
            return self
            
    
    def predict(self, messages):
        return self.predict_probabilities(messages).argmax(1).flatten()

    def predict_probabilities(self, messages):
        if self.english:
            messages = (messages['Translation'].values)
        else:
            messages = (messages['Message'].values)  
      
        n =  len(messages)
        X_test_input = self.convert_to_input(messages.astype(str))
        test_ds=tf.data.Dataset.from_tensor_slices((X_test_input[0],X_test_input[1],X_test_input[2],np.ones(n))).map(self.example_to_features).batch(12)
        
        self.model = self.model.from_pretrained(self.path)
        return self.model.predict(test_ds)
        

    def params_str(self):
        return ''

    def __str__(self):
        return '{}{}'.format(type(self).__name__, self.params_str())

    def evaluate(self, metric = 'accuracy', epochs = 3, validation_percent = 0.2, allow_import = True):
    
        X_train, y_train, X_test, y_test  = select_columns(x=tuple(self.imap_columns), y=self.target)
        
        
        self.fit(X_train, y_train, epochs =  epochs, validation_percent = validation_percent, allow_import = allow_import)
        
        pred = self.predict(X_test)
        y_test = self.Lab_Encoder.transform(y_test.to_numpy()).astype(float)
       
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        if metric == 'accuracy':
            return (pred == y_test).mean()
        elif metric == 'f-score':
            return f1_score(y_test, pred)