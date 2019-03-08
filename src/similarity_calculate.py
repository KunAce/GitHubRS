# This Python file is intended computing the similarity between the repos and users based on the topic probability

# Import necessary packages
import pandas as pd
from ast import literal_eval
from operator import add
from numpy import array

# Step 1: Load the repo-topic file

# topic reader
col_names_topics = ['repo_id',"topic_prop"]
dtypes_topics = {'repo_id':int}
reader_topic = pd.read_csv('../dataset/repo_topic_no_equal.csv', names=col_names_topics, dtype=dtypes_topics,
                         error_bad_lines=False, encoding='utf-8', skiprows = [0],#,15505],
                        converters = {"topic_prop": literal_eval})
print('Loaded repo-topic_prop table.')


# Step 2: Load the user-prop fie
# col_names_user_repo = ['user_id','repo_id']
# dtypes_user_repo = {'user_id':int,'repo_id':int}
# reader_user_repo = pd.read_csv('../dataset/user_repo_sample_100k.csv', names=col_names_user_repo, dtype=dtypes_user_repo,
#                          error_bad_lines=False, encoding='utf-8')
# print('Loaded user-repo table.')


# Step 2 Alternative : Load the user-prop file
def load_user_prop():
    col_names_user_prop = ['user_id','topic_prop']
    dtypes_user_prop = {'user_id':int}
    reader_user_prop = pd.read_csv('../dataset/user_prop_sample_100k_no_equal.csv',names = col_names_user_prop, dtype = dtypes_user_prop,
                                   error_bad_lines = False, encoding = 'utf-8', skiprows = [0],
                                   converters={"topic_prop": literal_eval})
    print('Loaded user-prop table (100k sample except zero,equal distribution).')
    return reader_user_prop


# Step **: Load the result from aggregate_user()
def load_result_average():
    col_names_result_average = ['topic_prop']
    reader_result_average = pd.read_csv('../dataset/prop_average_sample_100k.csv', names=col_names_result_average,
                                   error_bad_lines=False, encoding='utf-8', skiprows=[0],
                                   converters={"topic_prop": literal_eval})
    return reader_result_average

# Step 3: Aggregate all the user topic probability into one non-personalized entry
def aggregate_user():
    reader_user_prop = load_user_prop()
    prop_counter = len(reader_user_prop) # number of user-prop entries
    prop_list = [0] * 50
    for row in reader_user_prop.itertuples():
        prop_list = list(map(add, prop_list, row[2]))
    print('Finish the aggregation of all user-prop entries. Calculating the Average.')
    prop_list_average = [round(x / prop_counter, 4) for x in prop_list]
    print('Finish calculating the Average.')
    result = pd.DataFrame(columns=['topic_prop'])
    result = result.append({'topic_prop':prop_list_average}, ignore_index=True)
    result.to_csv('../dataset/prop_average_sample_100k.csv', index = False)

# Step 4: Compute Cosine Similarity
def compute_similarity():
    # Create a dataframe for result
    result_similarity = pd.DataFrame(columns = ['repo_id','similarity'])

    # Load the result from aggregate_user()
    reader_result_average = load_result_average()
    result_average = array(reader_result_average.iloc[0]['topic_prop'])

    # from sklearn.metrics.pairwise import cosine_similarity
    from numpy import dot
    from numpy.linalg import norm

    # Loop through the topic_prop table
    for row in reader_topic.itertuples():
        # print(row[2]) # row[2] is the topic_prop for each repo
        row_prop = array(row[2])

        # Method 1 Using sklearn
        # result = cosine_similarity([result_average],[row_prop]) # Method 1
        # print('result:',result)

        # Method 2 using numpy
        cos_sim = dot(result_average, row_prop) / (norm(result_average) * norm(row_prop))
        result_similarity = result_similarity.append({'repo_id':row[1],'similarity':cos_sim}, ignore_index= True)

    # Sorted the result based on similarity
    result_similarity.sort_values(by = ['similarity'], ascending = False, inplace = True)
    result_similarity.to_csv('../dataset/similarity_sample_100k.csv', index = False)
    print('Done')




# Run the program
# aggregate_user()
compute_similarity()

