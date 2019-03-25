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
reader = pd.read_csv('../dataset/desc_en.csv', encoding='utf_8_sig', names=col_names,skiprows = [0,15505])

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

print("Finish creating 'desc_lemmatized'.")


# Create the Dictionary and Corpus for Topic Modeling

# Create Dictionary
# id2word = corpora.Dictionary(desc_lemmatized)

# Create Corpus
desc_lemmatized_corpus = desc_lemmatized

# Term Document Frequency
# corpus = [id2word.doc2bow(desc_item) for desc_item in desc_lemmatized_corpus]
#
from sklearn.externals import joblib
#
# joblib.dump((id2word,desc_lemmatized,desc_lemmatized_corpus,corpus), "../model/desc_NMF_data.pkl")

# CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

desc_lemmatized_corpus = [' '.join(text) for text in desc_lemmatized_corpus]
vectorizer = CountVectorizer(stop_words = stop_words,  min_df= 1)
A = vectorizer.fit_transform(desc_lemmatized_corpus)
print( "Created %d X %d document-term matrix" % (A.shape[0], A.shape[1]) )

terms = vectorizer.get_feature_names()
print("Vocabulary has %d distinct terms" % len(terms))

# joblib.dump((A,terms), "../model/desc_NMF_CountVectorizer_test.pkl")

# Applying Term Weighting with TF-IDF

# NMF Topic Models

# Create the Topic Models and Test different Ks
kmin, kmax = 5 , 80

from sklearn import decomposition
topic_models = []
# try each value of k
for k in range(kmin,kmax+1,10):
    print("Applying NMF for k=%d ..." % k )
    # run NMF
    model = decomposition.NMF( init="nndsvd", n_components=k )
    W = model.fit_transform( A )
    H = model.components_
    # store for later
    topic_models.append( (k,W,H) )


# Build a Word Embedding
# import re
# class TokenGenerator:
#     def __init__( self, documents):
#         self.documents = documents
#         self.stopwords = stopwords
#         self.tokenizer = re.compile( r"(?u)\b\w\w+\b" )
#
#     def __iter__( self ):
#         print("Building Word2Vec model ...")
#         for doc in self.documents:
#             tokens = []
#             for tok in self.tokenizer.findall( doc ):
#                 if tok in self.stopwords:
#                     tokens.append( "<stopword>" )
#                 elif len(tok) >= 2:
#                     tokens.append( tok )
#             yield tokens

import gensim
# docgen = TokenGenerator( desc_lemmatized_corpus, stop_words )
# the model has 500 dimensions, the minimum document-term frequency is 20
w2v_model = gensim.models.Word2Vec(desc_lemmatized)#, size=500, min_count=20, sg=1)
print( "Model has %d terms" % len(w2v_model.wv.vocab) )

# Selecting the Number of Topics

def calculate_coherence( w2v_model, term_rankings ):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations( term_rankings[topic_index], 2 ):
            pair_scores.append( w2v_model.wv.similarity(pair[0], pair[1]) )
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)


import numpy as np
def get_descriptor( all_terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append( all_terms[term_index] )
    # output the results
    print(top_terms)
    return top_terms


from itertools import combinations
k_values = []
coherences = []
for (k,W,H) in topic_models:
    # Get all of the topic descriptors - the term_rankings, based on top 20 terms
    term_rankings = []
    print('the topic number of ', k, ':')
    for topic_index in range(k):
        print('Topic ',topic_index, ':')
        term_rankings.append( get_descriptor( terms, H, topic_index,20 ) )
    # Now calculate the coherence based on our Word2vec model
    k_values.append( k )
    coherences.append( calculate_coherence( w2v_model, term_rankings ) )
    print("K=%02d: Coherence=%.4f" % ( k, coherences[-1] ) )