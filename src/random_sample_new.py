# Import necessary packages
import pandas as pd
import random

# Use Pickle module to save the counter for iterating the user-repo list
import pickle

def load_user_list():
    try:
        with open('../dataset/random_sample_user_list.pkl', 'rb') as pickle_file_set:
            user_list_sample_set = pickle.load(pickle_file_set)
            print('Loaded user list sample set.')
    except IOError as e:
        print('File Not Found. Start to generate random set.')

        # user reader for random sampling
        col_names_user = ['user_id']
        dtypes_user = {'user_id': int}
        reader_user = pd.read_csv('../dataset/user_list.csv', names=col_names_user, dtype=dtypes_user,
                                  error_bad_lines=False, encoding='utf-8', skiprows=1)

        # Randomly sample 100k users from the user list
        user_list = reader_user['user_id'].values.tolist()
        user_list_sample = random.sample(user_list, 100000)

        # Turn the user_list_sample into a set
        user_list_sample_set = set(user_list_sample)
        print('Created user list set.')
        with open('../dataset/random_sample_user_list.pkl', 'wb') as pickle_file_set:
            pickle.dump(user_list_sample_set, pickle_file_set)
    return(user_list_sample_set)



# user repo reader
col_names_user_repo = ['user_id','repo_id']
dtypes_user_repo = {'user_id':int,'repo_id':int}
reader_user_repo = pd.read_csv('../dataset/user_repo_new_sorted.csv', names=col_names_user_repo, dtype=dtypes_user_repo,
                         error_bad_lines=False, encoding='utf-8',skiprows=1)
print('Loaded user-repo table.')

result = pd.DataFrame(columns=['user_id','repo_id'])

# load the counter for user-repo list
try:
    with open('../dataset/random_sample_counter.pkl', 'rb') as pickle_file:
        counter = pickle.load(pickle_file) # count on how many rows are already visited
        print('The counter is: ', counter)
except IOError as e:
    print('File Not Found. Start to initialize the counter.')
    counter = 0

# Find the sample user-repo pairs and output
def pick(result,counter, user_list_sample_set):
    for row in reader_user_repo.loc[counter:].itertuples():
        counter += 1
        print(counter)
        #row[1] user_id # row[2] repo_id
        if row[1] in user_list_sample_set: # if the user_id is in the sample set
            # result = result.append({'user_id':row[1],'repo_id':row[2]}, ignore_index = True)
            print('user_id:',row[1],'repo_id:',row[2])
            # result.to_csv('../dataset/user_repo_sample_100k_test.csv', index=False, mode = 'a')
            pd.DataFrame({'user_id':row[1],'repo_id':row[2]},index=[0]).to_csv('../dataset/user_repo_sample_100k.csv', index=False, mode='a',header = False)
        with open('../dataset/random_sample_counter.pkl', 'wb') as pickle_file:
            pickle.dump(counter, pickle_file)
    print("Finish")


pick(result, counter,load_user_list())