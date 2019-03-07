# This script is intended for showing the number of users in the table

import pandas as pd
from ast import literal_eval


def user_repo_check():
    # user repo reader
    col_names_user_repo = ['user_id','repo_id']
    dtypes_user_repo = {'user_id':int,'repo_id':int}
    reader_user_repo = pd.read_csv('../dataset/user_repo_sample_100k.csv', names=col_names_user_repo, dtype=dtypes_user_repo,
                             error_bad_lines=False, encoding='utf-8', header = None)

    print(len(reader_user_repo.groupby('user_id')))
    print('Done')


def user_prop_check():
    # user_prop reader
    col_names_user_prop = ['user_id', "topic_prop"]
    dtypes_user_prop = {'user_id': int}
    reader_user_prop = pd.read_csv('../dataset/user_prop_sample_100k.csv', names=col_names_user_prop, dtype=dtypes_user_prop,
                               error_bad_lines=False, encoding='utf-8',
                               converters={"topic_prop": literal_eval})

    # set the counters for equality and inequality
    equal_counter = 0
    inequal_counter = 0
    zero_counter = 0

    result = pd.DataFrame(columns=['user_id', 'topic_prop'])

    # Set the counters for equality and inequality

    def check_equal(equal_counter, inequal_counter, zero_counter, result_df):
        for index, row in reader_user_prop.iterrows():
            new_row = sorted(row['topic_prop'])
            if new_row[0] == new_row[49]:
                equal_counter += 1
                if new_row[0] == 0:
                    zero_counter += 1
            else:
                inequal_counter += 1
                result_df = result_df.append(row, ignore_index=True)
        print("Users with all equal distribution: ", equal_counter)
        print("Users with inequal distribution: ", inequal_counter)
        print("Users with zero distribution: ", zero_counter)
        result_df.to_csv('../dataset/user_prop_sample_100k_no_equal.csv', index=False)

    check_equal(equal_counter, inequal_counter, zero_counter, result)


# Run the program
user_prop_check()
