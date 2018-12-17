## Load the packages

import numpy as np
import pandas as pd
import re, nltk, spacy, gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
# %matplotlib inline

# Import the 'description' data

col_names = ['name','description']
reader = pd.read_csv('../dataset/desc_en.csv', encoding='utf_8_sig', names=col_names,skiprows = 1)

# # Prepare Stopwords
#
# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
# stop_words.extend(['from','re','edu','use'])
#
# Convert each row of description to list
desc_list = reader['description'].values.tolist()


# Tokenize words and Clean-up Text

def desc_to_words(desc_list):
    for desc_item in desc_list:
        yield(gensim.utils.simple_preprocess(str(desc_item),deacc=True)) # remove punctuations

desc_words=list(desc_to_words(desc_list))

# Lemmatization
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
desc_lemmatized = lemmatization(desc_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

vectorizer = CountVectorizer(analyzer='word',
                             min_df=10,                        # minimum occurrences of a word
                             stop_words='english',             # remove stop words
                             lowercase=True,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                            )

desc_lemmatized_str = [' '.join(text) for text in desc_lemmatized]

desc_vectorized = vectorizer.fit_transform(desc_lemmatized_str)


# Check the Sparsity

# Materialize the sparse data
desc_dense = desc_vectorized.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((desc_dense > 0).sum()/desc_dense.size)*100, "%")
#
# Build LDA Model
lda_model = LatentDirichletAllocation(n_topics=50,               # Number of topics
                                      max_iter=10,               # Max learning iterations
                                      learning_method='online',
                                      random_state=100,          # Random state
                                      batch_size=128,            # n docs in each learning iter
                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      n_jobs = -1,               # Use all available CPUs
                                     )

lda_output = lda_model.fit_transform(desc_vectorized)

print(lda_model)  # Model attributes

# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(desc_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(desc_vectorized))

# See model parameters
pprint(lda_model.get_params())

# GridSearch the best LDA model

# Define Search Param
search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(desc_vectorized)

GridSearchCV(cv=None, error_score='raise',
       estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7, learning_method=None,
             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
             mean_change_tol=0.001, n_components=10, n_jobs=1,
             n_topics=None, perp_tol=0.1, random_state=None,
             topic_word_prior=None, total_samples=1000000.0, verbose=0),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'n_topics': [10, 15, 20, 25, 30], 'learning_decay': [0.5, 0.7, 0.9]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)


# See the best topic model and its parameters

# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(desc_vectorized))


# Compare LDA Model Performance Scores

# Get Log Likelyhoods from Grid Search Output
n_topics = [10, 15, 20, 25, 30]
log_likelyhoods_5 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==0.5]
log_likelyhoods_7 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==0.7]
log_likelyhoods_9 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['learning_decay']==0.9]

# Show graph
plt.figure(figsize=(12, 8))
plt.plot(n_topics, log_likelyhoods_5, label='0.5')
plt.plot(n_topics, log_likelyhoods_7, label='0.7')
plt.plot(n_topics, log_likelyhoods_9, label='0.9')
plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel("Log Likelyhood Scores")
plt.legend(title='Learning decay', loc='best')
plt.show()


# See the dominant topic in each document

# Create Document - Topic Matrix
lda_output = best_lda_model.transform(desc_vectorized)

# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_topics)]

# index names
repo_names = ["Repo" + str(i) for i in range(len(desc_list))]

# Make the pandas dataframe
df_desc_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=repo_names)

# Get dominant topic for each document
dominant_topic = np.argmax(df_desc_topic.values, axis=1)
df_desc_topic['dominant_topic'] = dominant_topic

# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# Apply Style
df_desc_topics = df_desc_topic.head(15).style.applymap(color_green).applymap(make_bold)
df_desc_topics


# Review topics distribution across documents

df_topic_distribution = df_desc_topic['dominant_topic'].value_counts().reset_index(name="Num of Repos")
df_topic_distribution.columns = ['Topic Num', 'Num of Repos']
df_topic_distribution


# Visualize the LDA model with pyLDAvis
pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(best_lda_model, desc_vectorized, vectorizer, mds='tsne')
panel


# See the Topic’s keywords

# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(best_lda_model.components_)

# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames

# View
df_topic_keywords.head()


# Get the top 15 keywords each topic

# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Keyword '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
df_topic_keywords


# # Predict the topics for a new piece of description
#
# # Define function to predict topic for a given text document.
# nlp = spacy.load('en', disable=['parser', 'ner'])
#
# def predict_topic(text, nlp=nlp):
#     global sent_to_words
#     global lemmatization
#
#     # Step 1: Clean with simple_preprocess
#     mytext_2 = list(sent_to_words(text))
#
#     # Step 2: Lemmatize
#     mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#
#     # Step 3: Vectorize transform
#     mytext_4 = vectorizer.transform(mytext_3)
#
#     # Step 4: LDA Transform
#     topic_probability_scores = best_lda_model.transform(mytext_4)
#     topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :].values.tolist()
#     return topic, topic_probability_scores
#
# # Predict the topic
# mytext = ["Insert text here........"]
# topic, prob_scores = predict_topic(text = mytext)
# print(topic)


# # Cluster documents that share similar topics and plot
#
# # Construct the k-means clusters
# from sklearn.cluster import KMeans
# clusters = KMeans(n_clusters=15, random_state=100).fit_predict(lda_output)
#
# # Build the Singular Value Decomposition(SVD) model
# svd_model = TruncatedSVD(n_components=2)  # 2 components
# lda_output_svd = svd_model.fit_transform(lda_output)
#
# # X and Y axes of the plot using SVD decomposition
# x = lda_output_svd[:, 0]
# y = lda_output_svd[:, 1]
#
# # Weights for the 15 columns of lda_output, for each component
# print("Component's weights: \n", np.round(svd_model.components_, 2))
#
# # Percentage of total information in 'lda_output' explained by the two components
# print("Perc of Variance Explained: \n", np.round(svd_model.explained_variance_ratio_, 2))

# # Plot
# plt.figure(figsize=(12, 12))
# plt.scatter(x, y, c=clusters)
# plt.xlabel('Component 2')
# plt.xlabel('Component 1')
# plt.title("Segregation of Topic Clusters", )


# Get similar documents for any given piece of description
# #
# from sklearn.metrics.pairwise import euclidean_distances
#
# nlp = spacy.load('en', disable=['parser', 'ner'])
#
# def similar_documents(text, doc_topic_probs, documents = desc_list, nlp=nlp, top_n=5, verbose=False):
#     topic, x  = predict_topic(text)
#     dists = euclidean_distances(x.reshape(1, -1), doc_topic_probs)[0]
#     doc_ids = np.argsort(dists)[:top_n]
#     if verbose:
#         print("Topic KeyWords: ", topic)
#         print("Topic Prob Scores of Desc: ", np.round(x, 1))
#         print("Most Similar Repo's Probs:  ", np.round(doc_topic_probs[doc_ids], 1))
#     return doc_ids, np.take(documents, doc_ids)
#
# # Get similar documents
# mytext = ["Insert Text Here"]
# doc_ids, docs = similar_documents(text=mytext, doc_topic_probs=lda_output, documents = desc_list, top_n=1, verbose=True)
# print('\n', docs[0][:500])

