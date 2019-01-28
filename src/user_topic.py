# Import necessary packages
import pandas as pd
from ast import literal_eval

# user_repo reader
col_names_user_repo = ['user_id','repo_id']
dtypes_user_repo = {'user_id':int,'repo_id':int}
reader = pd.read_csv('../dataset/user_repo_new.csv',
                     skiprows = 1, names = col_names_user_repo,dtype=dtypes_user_repo,
                     error_bad_lines = False, encoding='utf-8', nrows = 1000)

# # topic reader
# col_names_topics = ['repo_id',"topic_prop"]
# dtypes_topics = {'repo_id':str}
# reader_topic = pd.read_csv('../dataset/repo_topic.csv', skiprows=1, names=col_names_topics, dtype=dtypes_topics,
#                          error_bad_lines=False, encoding='utf-8',
#                         converters = {"topic_prop": literal_eval})

# print(reader_topic.topic_prop[0])
# print(reader_topic)

# Find the specific userid
# Find all entries for 'userid'
def iterate_user(reader):
    # return a list of user
    user_reader = reader.drop_duplicates(subset = 'user_id', keep = "last")
    user_prop_list = pd.DataFrame(columns=['user_id','topic_prop']) # result list
    for index, row in user_reader.iterrows():
        user_prop_list.loc[index] = {'user_id': row['user_id'], 'topic_prop': find_user(row['user_id'])}
        user_prop_list.to_csv('../dataset/user_prop_test.csv', index = False, mode='a')
    # print(user_prop_list)



from operator import add
# Find a specific user_id
def find_user(userid):
    one_user = reader.loc[reader['user_id'] == userid] # multiple repo entries for one user
    # print(one_user)
    repo_counter = 0 #len(one_user.index) # for calculating average props
    prop_list = [0] * 50 # initialize the list with zeros
    for repo_row in one_user.iterrows():
        repo = repo_row[1][1] # the corresponding repo_id
        newlist = find_topic(repo)
        if len(newlist) != 0:
            repo_counter = repo_counter + 1
            prop_list = list(map(add , prop_list, newlist[0]))
    prop_list_new = [round(x / repo_counter, 4) for x in prop_list]
    return(prop_list_new) # Averaged list of topic props



# topic finder
def find_topic(repo):
    # topic reader
    col_names_topics = ['repo_id', "topic_prop"]
    dtypes_topics = {'repo_id': int}
    reader_topic = pd.read_csv('../dataset/repo_topic.csv', names=col_names_topics, dtype=dtypes_topics,
                               error_bad_lines=False, encoding='utf-8', skiprows = [0],
                               converters={"topic_prop": literal_eval})

    repo_topics = reader_topic.loc[reader_topic['repo_id'] == repo] # show the repo and its topic props list
    return(repo_topics.topic_prop.tolist())


# Run the code
iterate_user(reader)
