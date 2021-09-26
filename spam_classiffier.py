# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:59:44 2021

@author: t
"""

import pandas as pd
msg=pd.read_csv(r'E:\ml\NLP\smsspamcollection\SMSSpamCollection',sep='\t',names=['label','message'])
#dataset is tab seperated so sep='\t' divide words as columns of dataframe 
#dataset doesn't have columns hence label and message are two columns named

#Data cleaning:
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

corpus=[]
for i in range(0,len(msg)):
    review=re.sub('[^a-zA-Z]',' ',msg['message'][i])
    review=review.lower()
    review=review.split() #review now has list words seperated by space 
    review=[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review) # joining list of stemmed words into a sentence again
    corpus.append(review)
    
#creating a Bag of Words model:
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)# cv has most frequent words as columns
X=cv.fit_transform(corpus).toarray() #input data

y=pd.get_dummies(msg['label']) #ml algo will not understand texts present in labels, hence they need to be transformed into dummy variables
y=y.iloc[:,1].values #to avoid dummy variable trap we use only one column to represent presence of ham or spam
#spam is represented as 0, ham is represented as 1

#Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#training the model using Naive bayes classifier: naive bayes is efficient with nlp
from sklearn.naive_bayes import MultinomialNB #MultinomialNB works for any no. of classes
spam_detect_model=MultinomialNB().fit(X_train,y_train)
y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix #it provides 2X2 matrix of predicted values by the model
conf_matrix=confusion_matrix(y_test,y_pred)
#diagonal elements in matrix provide correctly predicted counts.So 946+152 are correctly predicted 

#To calculate accuracy:
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
#it has high performance as 98.48
#Performance tuning can be done by using lemmatization, TFIDF model  


