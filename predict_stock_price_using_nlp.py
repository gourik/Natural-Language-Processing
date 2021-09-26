# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:32:01 2021

@author: t
"""

#Predicting Stock Price Movement based on News Headline using NLP:
#Dataset considered is combination of World News and Stock Price shifts available on Kaggle.
#Data ranges from 2008 to 2016 and data from 2000 to 2008 was scrapped from Yahoo finance.
#Labels are based on Dow Jones industrial average stock index.
#Class 1:Stock price increased
#Class 0:Stock price stayed same or decreased

import pandas as pd

pd.pandas.set_option('display.max_columns',None)
df=pd.read_csv(r'E:\ml\NLP\Data.csv',encoding="ISO-8859-1")
df.head()

#Dividing data into train and test data:
train=df[df['Date']<'20150101']
test=df[df['Date']>'20141231']   
test['Label']
#removing punctuations:
data=train.iloc[:,2:27]
data.replace('[^a-zA-Z]',' ',regex=True,inplace=True)

#Renaming column names for easy access: 
list1=[str(i) for i in range(25)]
data.columns=list1
data.head(1)
#Converting headlines to lower case:
for i in list1:
  data[i]=data[i].str.lower()  
  
data.head(1)  
data.index
data.size
data.shape
#In order to apply BagOfWords or TFIDF, all the headlines present in each column should be 
#converted into a single paragraph.
headlines=[]
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

headlines    

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier 

#implementing BagofWords:
count_vector=CountVectorizer(ngram_range=(2,2))
train_dataset=count_vector.fit_transform(headlines)

#Implementing RandomForest Classifier:
random_forest=RandomForestClassifier(n_estimators=200,criterion='entropy')
random_forest.fit(train_dataset,train['Label'])

#Predicting for Test data:
test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset=count_vector.transform(test_transform)    
predictions=random_forest.predict(test_dataset)    

#Library to check accuracy:
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

matrix=confusion_matrix(test['Label'],predictions)
matrix
score=accuracy_score(test['Label'],predictions)
score
report=classification_report(test['Label'],predictions)
report
