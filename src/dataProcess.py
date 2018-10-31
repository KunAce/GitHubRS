# Import necessary packages
import pandas as pd
import setting

# Read the CSV file for 'Project'
#ignore the first row because it is a false entry

file=pd.read_csv('../dataset/projects.csv',nrows=1000,skiprows=1,names=['id','url','owner_id','name','description','language','created_at','forked_from','deleted','updated_at','unknown'])
dfproject=pd.DataFrame(file)
# print(dfproject.head())

# A table with tags of multiple topics for each repository
# file=pd.read_csv('../dataset/project_topics.csv',nrows=50)
# dftopics=pd.DataFrame(file)
# print(dftopics.head())

# Read the CSV file for 'watchers'
file=pd.read_csv('../dataset/watchers.csv',nrows=10000,names=['repo_id','user_id','created_at'])
dfwatchers=pd.DataFrame(file)
# print(dfwatchers.head())

# Group by 'id' and count 'user_id' for 'watchers' => Popularity for repository
dfwatcherscount=pd.DataFrame({'popularity':dfwatchers.groupby(['repo_id']).size()}).reset_index()
# print(dfwatcherscount.head())

# Merge the popularity to 'projects' table
setting.init()
setting.res=pd.merge(dfproject,dfwatcherscount,left_on='id',right_on='repo_id',how='left')
print(setting.res.head())

# print(df.tail())

# pd.set_option('display.max_columns',12)
# pd.set_option('display.max_colwidth',20)
# pd.set_option('display.width',-1)