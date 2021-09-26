# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:33:54 2021

@author: t
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords #stopWords is library used to remove non-significant words in paragraph which don't 
# contribute much in analysing part such as 'of', 'and', 'them' etc.  

paragraph ="""In 3000 years of our history, people from all over the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British, the French, the Dutch, all of them came and looted us, took over what was ours. 
           Yet, we have not done this to any other nation. We have not conquered anyone. 
           We have not grabbed their land, their culture, their history and tried to enforce our way of life on them. 
           Why? Because we respect the freedom of others.That is why my first vision is that of FREEDOM.
           I believe that India got its first vision of this in 1857, when we started the war of Independence. It is this freedom that we must protect and nurture and build on. If we are not free, no one will respect us.
           My second vision for India’s DEVELOPMENT. For fifty years we have been a developing nation. It is time we see ourselves as a developed nation. We are among top five nations of the world in terms of GDP. We have 10 per cent growth rate in most areas. Our poverty levels are falling. Our achievements are being globally recognised today. Yet we lack the self-confidence to see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
           I have a THIRD vision. India must stand up to the world. Because I believe that, unless India stands up to the world, no one will respect us. Only strength respects strength. We must be strong not only as a military power but also as an economic power. Both must go hand-in-hand. My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Department of Space, Professor Satish Dhawan, who succeeded him and Dr.Brahm Prakash, the father of nuclear material. I was lucky to have worked with all three of them closely and consider this the great opportunity of my life.
           I see four milestones in my career:"""

#First step in Tokenization is converting paragraph into Sentences:
sentences=nltk.sent_tokenize(paragraph)
stemmer=PorterStemmer() #for stemming purpose

# stopwords.words('english') #It gives a list of words which are stopwords
# stopwords.words('german')

#Stemming:This also includes removing stopwords:
for i in range(len(sentences)):
    words=nltk.word_tokenize(sentences[i])
    words=[stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i]=' '.join(words)
#now sentences do not have any stopwords and they are stemmed also.
#But stemming produces a stem words which dont have any meaning. So we need to do Lemmatization to have meaningful stem words
