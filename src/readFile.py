# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Read the CSV file for 'Project'
#ignore the first row because it is a false entry

file=pd.read_csv('../dataset/projects.csv',nrows=1000,skiprows=1,names=['id','url','owner_id','name','description','language','created_at','forked_from','deleted','updated_at','unknown'])
dfproject=pd.DataFrame(file)
# print(dfproject.head())

# A table with tags of multiple topics for each repository
# file=pd.read_csv('../dataset/project_topics.csv',nrows=50)
# dftopics=pd.DataFrame(file)
# print(dftopics.head())

# Read the CSV file for 'watchers'
file=pd.read_csv('../dataset/watchers.csv',nrows=10000,names=['repo_id','user_id','created_at'])
dfwatchers=pd.DataFrame(file)
# print(dfwatchers.head())

# Group by 'id' and count 'user_id' for 'watchers' => Popularity for repository
dfwatcherscount=pd.DataFrame({'popularity':dfwatchers.groupby(['repo_id']).size()}).reset_index()
# print(dfwatcherscount.head())

# Merge the popularity to 'projects' table
res=pd.merge(dfproject,dfwatcherscount,left_on='id',right_on='repo_id',how='left')
print(res.head())

#Source:
#Modified from https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

# Topic Modelling

# Extract 'description' from the 'project' table


# Text Cleaning


#return a list of tokens
import spacy
#spacy.load('en_core_web_sm')
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

# Find the meanings of words by using NLTK's Wordnet

import nltk

nltk.download('wordnet')
from nltk.corpus import wordnet as wn


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


# Filter stop words
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))


# Function for preparing the text for topic modelling
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

# Convert text data from 'description'
import random
text_data = []
resDesc=res.loc[:,['description']]
resDesc = resDesc[pd.notnull(resDesc['description'])]

# #temporarily fill the NAN
# resDesc.fillna("")

print(resDesc)
for row in resDesc.iterrows():
    tokens = prepare_text_for_lda(row[1].values[0])
    # if random.random() > .99:
    print(tokens)
    text_data.append(tokens)


# LDA with Gensim
from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')


# Find topics in the data

import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)


# pyLDAvis

dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.show(lda_display)


# print(df.tail())

# pd.set_option('display.max_columns',12)
# pd.set_option('display.max_colwidth',20)
# pd.set_option('display.width',-1)