# This python script is intended for generating the recommendation list based on the similarity between the training set and the repo list

# Import necessary packages
import pandas as pd
from ast import literal_eval
from operator import add
from numpy import array

# Load the repo prop table
col_names_topics = ['repo_id',"topic_prop"]
dtypes_topics = {'repo_id':int}
reader_topic = pd.read_csv('../dataset/repo_topic.csv', names=col_names_topics, dtype=dtypes_topics,
                         error_bad_lines=False, encoding='utf-8', skiprows = [0,15505],
                        converters = {"topic_prop": literal_eval})
print('Loaded repo-topic_prop table.')

# Step 1: Load the training set

def load_user_prop_train():
    col_names_user_prop_train = ['user_id','topic_prop']
    dtypes_user_prop_train = {'user_id':int}
    reader_user_prop_train = pd.read_csv('../dataset/user_prop_over10_train.csv',names = col_names_user_prop_train, dtype = dtypes_user_prop_train,
                                   error_bad_lines = False, encoding = 'utf-8',
                                   converters={"topic_prop": literal_eval})
    print('Loaded user-prop table from Train Set.')
    return reader_user_prop_train


# Step 2: Aggregate all the user props into one vector to generate a non-personalized user profile
def generate_user_profile():
    reader_user_prop = load_user_prop_train()
    prop_counter = len(reader_user_prop)  # number of user-prop entries
    prop_list = [0] * 50
    for row in reader_user_prop.itertuples():
        prop_list = list(map(add, prop_list, row[2]))
    print('Finish the aggregation of all user-prop entries. Calculating the Average.')
    prop_list_average = [round(x / prop_counter, 4) for x in prop_list]
    print('Finish calculating the Average.')
    result = pd.DataFrame(columns=['topic_prop'])
    result = result.append({'topic_prop': prop_list_average}, ignore_index=True)
    result.to_csv('../dataset/prop_average_train.csv', index=False)


# Step 3: Calculate the similarity between the user profile and each repo
def compute_similarity():
    # Create a dataframe for result
    result_similarity = pd.DataFrame(columns = ['repo_id','similarity'])

    # Load the result from aggregate_user()
    col_names_result_average = ['topic_prop']
    reader_result_average = pd.read_csv('../dataset/prop_average_train.csv', names=col_names_result_average,
                                   error_bad_lines=False, encoding='utf-8', skiprows=[0],
                                   converters={"topic_prop": literal_eval})

    result_average = array(reader_result_average.iloc[0]['topic_prop'])

    # from sklearn.metrics.pairwise import cosine_similarity
    from numpy import dot
    from numpy.linalg import norm

    # Loop through the topic_prop table
    for row in reader_topic.itertuples():
        # print(row[2]) # row[2] is the topic_prop for each repo
        row_prop = array(row[2])

        # cosine similarity
        cos_sim = dot(result_average, row_prop) / (norm(result_average) * norm(row_prop))
        result_similarity = result_similarity.append({'repo_id':row[1],'similarity':cos_sim}, ignore_index= True)

    # Sorted the result based on similarity
    result_similarity.sort_values(by = ['similarity'], ascending = False, inplace = True)
    result_similarity.to_csv('../dataset/similarity_over10_train.csv', index = False)
    print('Finish computing similarity.')


# # Step 4: Generate the recommendation list
def generate_recommendation_list():
    # Load the result from the sorted similarity table
    col_names_similarity = ['repo_id','similarity']
    reader_similarity = pd.read_csv('../dataset/similarity_over10_train.csv', names=col_names_similarity,
                                        error_bad_lines=False, encoding='utf-8', skiprows=[0])

    # recommend the top 1000 repos
    reader_similarity = reader_similarity.iloc[0:10000]['repo_id']
    set_recommend = set(reader_similarity)

    # # Set a counter to calculate the precision rate
    # counter = 0

    # Load test set
    col_names_user_repo_test = ['user_id', 'repo_id']
    dtypes_user_repo_test = {'user_id': int, 'repo_id':int}
    reader_user_repo_test = pd.read_csv('../dataset/user_repo_over10_test.csv', names=col_names_user_repo_test,
                                        dtype=dtypes_user_repo_test,
                                        error_bad_lines=False, encoding='utf-8')
    grouped = reader_user_repo_test.groupby('repo_id').count().reset_index()
    grouped = grouped.sort_values('repo_id', ascending= False)

    set_test = set(grouped.head(10000)['repo_id'])
    print(len(set_recommend.intersection(set_test)))
    print("Done.")


# # Step 5: Evaluate against the test set
# def evaluate():

# Run the program
# generate_user_profile()
# compute_similarity()
generate_recommendation_list()