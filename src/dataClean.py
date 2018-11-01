# Import necessary packages
import pandas as pd
import numpy as np

def cleanProject():
    # Read the CSV file for 'Project'
    #ignore the first row because it is a false entry

    file=pd.read_csv('../dataset/projects.csv', nrows=5000, skiprows=1, names=['id','url','owner_id','name','description','language','created_at','forked_from','deleted','updated_at','unknown'])
    dfproject=pd.DataFrame(file)

    # 1) clean 'forked from'
    # dfproject = dfproject[dfproject['forked_from'].notnull]

    # 2) clean 'deleted'
    dfproject = dfproject[dfproject['deleted'] == 0]

    # 3) clean 'unknown' column
    dfproject = dfproject.drop('unknown', 1)

    # Output the new CSV file
    dfproject.to_csv('../dataset/project_test.csv', sep=',')



