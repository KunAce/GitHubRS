# Import necessary packages
import pandas as pd
import numpy as np


# Count popularity by Grouping
def count(reader):

    # Group by 'id' and count 'user_id' for 'watchers' => Popularity for repository
    watcher_count = pd.DataFrame({'popularity':reader.groupby(['repo_id']).size()}).reset_index()

    # Output the new CSV file
    watcher_count.to_csv('../dataset/popularity_count.csv', sep=',')


# Read the csv file and process

col_names = ['repo_id','user_id','created_at']
reader = pd.read_csv('../dataset/watchers.csv',names=col_names)
count(reader)
