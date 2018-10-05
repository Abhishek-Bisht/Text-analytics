# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:58:37 2018

@author: u343230
"""
'''
-----------  Downloading all the required packages---------------
'''
# for lemmatization
import spacy
'''
Other Spacy models which can be imported based on model performance
en_core_web_md, en_core_web_lg, en_vectors_web_lg
'''
import en_core_web_sm      #Basic Spacy model for English
nlp = en_core_web_sm.load()


import nltk

#nltk.download('stopwords')

import re
import numpy as np
import pandas as pd
from pprint import pprint
#import pyemojify

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# based on input data
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

#data = pd.read_csv("C:/Abhishek/work/Twitter POC/Apple_self_drive_car.csv",encoding='latin-1')

df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')

print(df.target_names.unique())
print(df.target.unique())

df.head()

# Convert to list
data = df.content.values.tolist()

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

pprint(data[:1])

data_words=[]
for sentence in data:
    datawords=gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations
    data_words.append(datawords)

print(data_words[:1])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(bigram_mod[data_words[1]])


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#trial=data_words[1]
#trial_1=[[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in trial]

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
#nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

# Human readable format of corpus (term-frequency)
#[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

import os

os.environ['MALLET_HOME'] = 'C:\\Abhishek\\work\\Twitter_POC\\mallet-2.0.8'

from gensim.models.wrappers import LdaMallet
# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
mallet_path = 'C:\\Abhishek\\work\\Twitter_POC\\mallet-2.0.8\\bin\\mallet' # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)

# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)

'''
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaMallet

path_to_mallet_binary = "C:/Abhishek/work/Twitter_POC/mallet-2.0.8/bin/mallet"
model = LdaMallet(path_to_mallet_binary, corpus=common_corpus, num_topics=20, id2word=common_dictionary)
vector = model[common_corpus[0]]
'''
'''
def GetNer(ner_model):
   command = 'java -Xmx256m -cp %s/mallet-2.0.6/lib/mallet-deps.jar:%s/mallet-2.0.6/class 
       cc.mallet.fst.SimpleTaggerStdin --weights sparse --model-file %s/models/ner/%s' 
       % (BASE_DIR, BASE_DIR, BASE_DIR, ner_model)
   return subprocess.Popen(command, shell=True, close_fds=True,  
       stdin=subprocess.PIPE, stdout=subprocess.PIPE) 

ner.stdin.write(("\t".join(seq_features) + "\n").encode('utf8'))  
ner.stdout.readline().rstrip('\n').strip(' ')  
'''


 