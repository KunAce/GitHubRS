# Import necessary packages
import pandas as pd



# Read the CSV file for 'Project'
#ignore the first row because it is a false entry

file = pd.read_csv('../dataset/projects.csv',nrows=1000,skiprows=1,names=['id','url','owner_id','name','description','language','created_at','forked_from','deleted','updated_at','unknown'])
dfproject = pd.DataFrame(file)

dfproject.to_csv('../dataset/project_short.csv', sep=',')

print(dfproject.head())
