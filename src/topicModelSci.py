import setting
import pandas as pd
import dataProcess
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn


dataProcess.run()

# get the raw 'description' from new 'project' table
resDesc = setting.res.loc[:,['description']]
resDesc = resDesc[pd.notnull(resDesc['description'])]

# set up variables

n_features=1000
n_topics=10
n_top_wrods = 20

# feature_extraction and vectorization

tf_vectorizer = CountVectorizer(strip_accents='unicode', max_features=n_features, stop_words='english')

tf = tf_vectorizer.fit_transform(resDesc['description'])

# applying LDA

lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50, learning_method='online', learning_offset=50., random_state=0)


lda.fit(tf)


# show the TopN keywords in each topic


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                       for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()


tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_wrods)


# pyLDAvis to show graph

graph = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
pyLDAvis.show(graph)