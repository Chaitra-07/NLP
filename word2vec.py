# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:27:07 2021

@author: achai
"""

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
import urllib 
import re
import bs4 as bs 
from gensim.models import Word2Vec

#importing the data
source = urllib.request.urlopen('https://en.wikipedia.org/wiki/Global_warming').read()

# Parsing the data/ creating BeautifulSoup object
soup = bs.BeautifulSoup(source,'lxml')

text = ""
for paragraph in soup.find_all('p'):
    text+= paragraph.text
    
# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\W',' ',text)
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)

words = model.wv.vocab

# Finding Word Vectors
vector = model.wv['global']

# Most similar words
similar = model.wv.most_similar('global')


