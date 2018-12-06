# The code below is modified based on the research objectives of my own project and is from the source:
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

# Text Pre-Processing

import nltk;
# nltk.download('stopwords')

# Import Packages
import re
import numpy as np
import pandas as pd
from pprint import pprint

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

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Prepare Stopwords

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from','re','edu','use'])

# Import the 'description' data

col_names = ['name','description']
reader = pd.read_csv('../dataset/desc_en.csv', encoding='utf_8_sig', names=col_names,skiprows = 1, nrows = 1000)

# Remove non-english by regular expressions
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

# # See trigram example
# print(trigram_mod[bigram_mod[desc_words[20]]])


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


# Create the Dictionary and Corpus for Topic Modeling

# Create Dictionary
id2word = corpora.Dictionary(desc_lemmatized)

# Create Corpus
desc_lemmatized_corpus = desc_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(desc_item) for desc_item in desc_lemmatized_corpus]


# Building the Topic Model LDA

# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=50,
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100,
#                                            passes=10,
#                                            alpha='auto',
#                                            per_word_topics=True)
#
#
# # Print the Keyword in the 10 topics
# pprint(lda_model.print_topics())
# doc_lda = lda_model[corpus]
#
#
# # Compute Model Perplexity and Coherence Score
#
# # Compute Perplexity
# print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
#
# # Compute Coherence Score
# coherence_model_lda = CoherenceModel(model=lda_model, texts=desc_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ', coherence_lda)


# Visualize the topics

# display = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# pyLDAvis.show(display)


# Building LDA Mallet Model

mallet_path='../package/mallet/bin/mallet'

ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=50, id2word=id2word)
#
# # Show Topics
# pprint(ldamallet.show_topics(formatted=False))
#
# # Compute Coherence Score
# coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=desc_lemmatized, dictionary=id2word, coherence='c_v')
# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
# print('\nCoherence Score: ', coherence_ldamallet)

#
# def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
#     """
#     Compute c_v coherence for various number of topics
#
#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     limit : Max num of topics
#
#     Returns:
#     -------
#     model_list : List of LDA topic models
#     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
#     """
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())
#
#     return model_list, coherence_values
#
#
#
# model_list, coherence_values = compute_coherence_values(dictionary=id2word,
#                                                         corpus=corpus,
#                                                         texts=desc_lemmatized,
#                                                         start=40, limit=60, step=5)
#
#
# # Show graph
# limit=60; start=40; step=5;
# x = range(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()
# #
# #
# # # Print the coherence scores
# # for m, cv in zip(x, coherence_values):
# #     print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
#
# # Select the model and print the topics
# optimal_model = model_list[1]
# model_topics = optimal_model.show_topics(formatted=False)
# pprint(optimal_model.print_topics(num_words=10))
#
#
#
# # # Finding the dominant topic in each description
#
def format_topics_description(ldamodel=ldamallet, corpus=corpus, texts=desc_lemmatized):
    # Init output
    desc_topics_df = pd.DataFrame()

    # Get main topic in each description
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Contribution and Keywords for each description
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                desc_topics_df = desc_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    desc_topics_df.columns = ['Dominant_Topic', 'Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    desc_topics_df = pd.concat([desc_topics_df, contents], axis=1)
    return(desc_topics_df)


df_topic_desc_keywords = format_topics_description(ldamodel=ldamallet, corpus=corpus, texts=desc_lemmatized) #change optimal_model
#
# # Format
# df_dominant_topic = df_topic_desc_keywords.reset_index()
# df_dominant_topic.columns = ['Repo_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Description']

# Show
# print(df_dominant_topic.head(10))


# Topic distribution across descriptions

# Number of descriptions for Each Topic
topic_counts = df_topic_desc_keywords['Dominant_Topic'].value_counts()

# Percentage of descriptions for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_desc_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Repo', 'Perc_Description']

# Show
df_dominant_topics
