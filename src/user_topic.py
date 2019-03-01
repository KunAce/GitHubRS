# Import necessary packages
import pandas as pd
from ast import literal_eval

# Use Pickle module to save the counter for iterating the user list
import pickle


# user_repo reader
col_names_user_repo = ['user_id','repo_id']
dtypes_user_repo = {'user_id':int,'repo_id':int}
reader = pd.read_csv('../dataset/user_repo_sample_100k.csv',
                    names = col_names_user_repo,dtype=dtypes_user_repo,
                     error_bad_lines = False, encoding='utf-8')

# topic reader
col_names_topics = ['repo_id',"topic_prop"]
dtypes_topics = {'repo_id':int}
reader_topic = pd.read_csv('../dataset/repo_topic.csv', names=col_names_topics, dtype=dtypes_topics,
                         error_bad_lines=False, encoding='utf-8', skiprows = [0,15505],
                        converters = {"topic_prop": literal_eval})

# Turn the reader_topic into a dict
reader_topic = reader_topic.set_index(['repo_id']).to_dict(orient = 'index')
print('Created Topic Dictionary.')


# print(reader_topic.topic_prop[0])
# print(reader_topic)


# Create a user list for iteration
def user_group():
    user_reader = reader.drop_duplicates(subset='user_id', keep="last")
    user_reader.drop(['repo_id'], axis = 1, inplace = True)
    user_reader.to_csv('../dataset/user_list_sample_100k.csv', index = False)


# Find the specific userid

# Find all entries for 'userid'
def iterate_user():
    # return a list of user

    # load the counter
    try:
        with open('../dataset/user_counter.pkl', 'rb') as pickle_file:
            counter = pickle.load(pickle_file)
    except IOError as e:
        print('File Not Found. Set counter to 0.')
        counter = 0

    # the user list
    reader_user_list = pd.read_csv('../dataset/user_list_sample_100k.csv', names = ['user_id'],
                                   error_bad_lines=False, encoding='utf-8', skiprows = 1)

    # user_prop_list = pd.DataFrame(columns=['user_id', 'topic_prop'])  # result list

    for index ,row in reader_user_list.loc[counter:].itertuples():
        # user_prop_list = pd.DataFrame(columns=['user_id', 'topic_prop'])  # result list
        # # user_prop_list.loc[index] = {'user_id': row['user_id'], 'topic_prop': find_user(row['user_id'])}
        # user_prop_list = user_prop_list.append({'user_id': row['user_id'], 'topic_prop': find_user(row['user_id'])},ignore_index=True)
        # user_prop_list.to_csv('../dataset/user_prop_sample_100k.csv', index = False , mode = 'a', header = False)

        # row[1] user_id # row[2] repo_id
        pd.DataFrame({'user_id': row, 'topic_prop': find_user(row)}).to_csv('../dataset/user_prop_sample_100k_test.csv',index=False,mode='a',header = False)

        #the counter to record how many users are already calculated
        counter = counter + 1
        print(counter)
        with open('../dataset/user_counter.pkl', 'wb') as pickle_file:
            pickle.dump(counter, pickle_file)
    print("Done")



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
        if newlist is not None:
            repo_counter = repo_counter + 1
            prop_list = list(map(add , prop_list, newlist['topic_prop']))
    if repo_counter != 0:
        prop_list_new = [round(x / repo_counter, 4) for x in prop_list]
        return prop_list_new  # Averaged list of topic props
    else:
        return prop_list


# topic finder
def find_topic(repo):
    if repo in reader_topic:
        return(reader_topic[repo])
    else:
        return


# Run the code
iterate_user()

# user_group()
