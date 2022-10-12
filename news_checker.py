#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import requests
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB


# In[19]:


def is_fake(news):
    
    df = pd.read_csv('train.csv')
    df = df.dropna()
    
    df['Whole'] = df['title'] + df['text']
    
    labels = df.label
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    
    
    tfidf_train = tfidf_vectorizer.fit_transform(df['Whole']) 
    tfidf_test=tfidf_vectorizer.transform([news])
    
    pac = PassiveAggressiveClassifier(max_iter = 50)
    pac.fit(tfidf_train,labels)

    y_pred = pac.predict(tfidf_test)
    
    if y_pred[0] == 0:
        return 'FAKE NEWS'
    else:
        return "REAL NEWS"


# In[23]:


news = input()
print(is_fake(news))


# In[ ]:




