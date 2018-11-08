# Import necessary packages
import pandas as pd

def cleanProject():
    # Read the CSV file for 'Project'
    #ignore the first row because it is a false entry

    file=pd.read_csv('../dataset/projects.csv', nrows=2000, skiprows=1, names=['id','url','owner_id','name','description','language','created_at','forked_from','deleted','updated_at','unknown'])
    dfproject=pd.DataFrame(file)

    # 1) clean 'forked from'
    # Only not 'forked_from' records are kept (with '\N')
    dfproject = dfproject[dfproject['forked_from'].values.astype(str) == '\\N']

    # 2) clean 'deleted'
    dfproject = dfproject[dfproject['deleted'] == 0]

    # 3) clean 'unknown' column
    dfproject = dfproject.drop('unknown', 1)

    # 4) re-format the 'url' column to make the links usable
    dfproject['url'].replace({'api.': ''}, inplace=True, regex=True)
    dfproject['url'].replace({'repos/': ''}, inplace=True, regex=True)


    # Output the new CSV file
    dfproject.to_csv('../dataset/project_test_2.csv', sep=',')





cleanProject()
