import pandas as pd
import numpy as np

from nltk.corpus import stopwords
import nltk

from gensim.models import ldamodel
import gensim.corpora

import pickle

#Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
# %matplotlib inline

# Prepare Stopwords

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from','re','edu','use'])


# Import the 'description' data

col_names = ['name','description']
reader = pd.read_csv('../dataset/desc_en.csv', encoding='utf_8_sig', names=col_names,skiprows = 1, nrows = 1000)

# Convert each row of description to list
desc_list = reader['description'].values.tolist()


# Tokenize words and Clean-up Text

def desc_to_words(desc_list):
    for desc_item in desc_list:
        yield(gensim.utils.simple_preprocess(str(desc_item),deacc=True)) # remove punctuations

desc_words=list(desc_to_words(desc_list))


# Creating Bigram and Tirgram Models

bigram = gensim.models.Phrases(desc_words, min_count=5, threshold=100) # higher threshold fewer phrases
trigram = gensim.models.Phrases(bigram[desc_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod= gensim.models.phrases.Phraser(bigram)
trigram_mod=gensim.models.phrases.Phraser(trigram)


# Remove stopwords and Lemmatize

# Define functions for stopwords,bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Remove Stop Words
desc_words_nostops = remove_stopwords(desc_words)

# Form Bigrams
desc_words_bigrams = make_bigrams(desc_words_nostops)

nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
desc_lemmatized = lemmatization(desc_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


# Vectorization => TF-IDF
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer()
# A = vectorizer.fit_transform(desc_lemmatized)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

desc_lemmatized_str = [' '.join(text) for text in desc_lemmatized]
A = vectorizer.fit_transform(desc_lemmatized_str)

# Access the list of all terms and an associated dictionary (vocabulary_)
# which maps each unique term to a corresponding column in the matrix.
terms = vectorizer.get_feature_names()
# len(terms)

vocab = vectorizer.vocabulary_
# vocab["world"]

# Number of Topics
k = 50

from sklearn import decomposition
model = decomposition.NMF(n_components=k, init="nndsvd")
W = model.fit_transform(A)
H = model.components_

# import numpy as np
top_indices = np.argsort( H[topic_index,:] )[::-1]
top_terms = []
for term_index in top_indices[0:top]:
    top_terms.append( terms[term_index] )


# top_indices = np.argsort( W[:,topic_index] )[::-1]
top_documents = []
for doc_index in top_indices[0:top]:
    top_documents.append( desc_lemmatized_str[doc_index] )

