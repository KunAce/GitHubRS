# Import necessary packages
import pandas as pd
from ast import literal_eval
from operator import add

# user_repo reader
col_names_user_repo = ['user_id','repo_id']
dtypes_user_repo = {'user_id':int,'repo_id':int}
reader = pd.read_csv('../dataset/user_repo_over10_test.csv',
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


# Find all entries for 'userid'
def iterate_user():
    user_group = reader.groupby('user_id')

    for user_id, group in user_group:
        repocounter = 0 # for calculating average props
        prop_list = [0] * 50  # initialize the list with zeros

        # Find and calculate the average of topic probability
        for repo_row in group.itertuples():
            repo = repo_row[2]  # the corresponding repo_id
            newlist = find_topic(repo)
            if newlist is not None:
                repo_counter = repo_counter + 1
                prop_list = list(map(add, prop_list, newlist['topic_prop']))

        if repo_counter != 0:
            prop_list = [round(x / repo_counter, 4) for x in prop_list] # Averaged list of topic props

        result = pd.DataFrame(columns=['user_id', 'topic_prop'])
        result = result.append({'user_id': user_id, 'topic_prop': prop_list},ignore_index = True)
        result.to_csv('../dataset/user_prop_over10_test.csv', index = False, mode = 'a', header = False)
    print("Done")


# topic finder
def find_topic(repo):
    if repo in reader_topic:
        return(reader_topic[repo])
    else:
        return

# Run the code
iterate_user()

