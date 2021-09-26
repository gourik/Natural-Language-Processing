# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 15:53:20 2021

@author: t
"""

import pandas as pd
df=pd.read_csv(r'E:\ml\NLP\train_fake_news.csv')
df.head()
df.shape
#Getting independent feature:
X=df.iloc[:,0:4]
#Alternatively we can drop feature 'label' from dataframe:
#X=df.drop('label',axis=1)
X

#Retrieving dependent feature:
y=df.iloc[:,-1]
# y=df['label']
y

from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
df=df.dropna()
df.shape
messages=df.copy()
messages.reset_index(inplace=True)
#messages['title'][6]
#messages.drop('level_0',axis=1)
messages.head(10)
len(messages)

#Applying stemming:
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
ps=PorterStemmer()
corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['text'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X=tfidf.fit_transform(corpus).toarray()
X.shape

y=messages['label']
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)

tfidf.get_feature_names()[:20]
tfidf.get_params() #it shows what features have been implemented on tfidf model:

#To view how vectors have been formed, we can construct dataframe from X_train:    
count_df=pd.DataFrame(X_train,columns=tfidf.get_feature_names())
count_df

import matplotlib.pyplot as plt
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matix',cmap=plt.cm.Blues):

    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes)) 
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm=cm.astype('float') /cm.sum(axis=1)[:,np.newaxis]
        print("Normalised confusion matrix")
    else:
        print("Confusion matrix, without normalisation")
        
    thresh=cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
    
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

#Multinomial Naive Bayes Algorithm:
from sklearn.naive_bayes import MultinomialNB #this works well for multi catogorial features
classifier=MultinomialNB()

from sklearn import metrics
import numpy as np
import itertools

classifier.fit(X_train,y_train)
pred=classifier.predict(X_test)
score=metrics.accuracy_score(y_test,pred)
score
print("accuracy: %0.3f" %score) 
cm=metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix(cm,classes=['FAKE','REAL'])
#Cross validation is not done as the data is huge, which requires high configurated machine,hence accuracy will be bit low


#Passive Aggressive Classifier Algorithm:It also works very efficiently on text data:
from sklearn.linear_model import PassiveAggressiveClassifier
linear_classifier=PassiveAggressiveClassifier(n_iter_no_change=50)
linear_classifier.fit(X_train,y_train)    
pred=linear_classifier.predict(X_test)
score=metrics.accuracy_score(y_test,pred)
print("accuracy: %0.3f" %score)
cm=metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix(cm, classes=['Fake Data', 'Real Data'])

#Multinomial Classifier with Hyperparameter tuning:
classifier=MultinomialNB(alpha=0.1)
previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train, y_train)
    y_pred=sub_classifier.predict(X_test)
    score=metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, score: {}".format(alpha,score))
    
    
#To extract feature names:
feature_names=tfidf.get_feature_names()

classifier.coef_[0]

#most real:(less -ve values):first 30 values:
sorted(zip(classifier.coef_[0],feature_names),reverse=True)[:30]

#most fake:first 10 values:
sorted(zip(classifier.coef_[0],feature_names))[:10]

#Hashing Vectorizer:
hs_vectorizer=HashingVectorizer(n_features=5000,non_negative=True)
X=hs_vectorizer.fit_transform(corpus).toarray()
X.shape
X

#Divide data into Train Test Split:
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)
    
classifier.fit(X_train,y_train)
pred=classifier.predict(X_test)
score=metrics.accuracy_score(y_test,pred)
score
print("accuracy: %0.3f" %score) 
cm=metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix(cm,classes=['FAKE','REAL'])
#Cross validation is not done as the data is huge, which requires high configurated machine,hence accuracy will be bit low

    