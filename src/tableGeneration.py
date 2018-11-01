# Import necessary packages
import pandas as pd
import numpy as np



# Read the CSV file for 'watchers'
file = pd.read_csv('../dataset/watchers.csv', nrows=10000, names=['repo_id', 'user_id', 'created_at'])
dfwatchers = pd.DataFrame(file)

# Group by 'id' and count 'user_id' for 'watchers' => Popularity for repository
dfwatcherscount =pd.DataFrame({'popularity' :dfwatchers.groupby(['repo_id']).size()}).reset_index()

# Output the new CSV file
dfwatcherscount.to_csv('../dataset/watcher_count.csv', sep=',')