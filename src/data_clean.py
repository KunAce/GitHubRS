# Import necessary packages
import pandas as pd
import numpy as np


# Data Cleaning
def clean(df_chunk):
    # Read the CSV file for 'Project'
    #ignore the first row because it is a false entry

    # 1) clean duplicates
    df_chunk.drop_duplicates()

    # 2) clean entry with empty description
    # print(df_chunk['description'].isnull().value_counts())
    df_chunk['description'].replace({'\\N': np.nan}, inplace = True)
    df_chunk.dropna(subset=['description'], inplace = True) # NaN data
    # print(df_chunk['description'].isnull().value_counts())

    # 1) clean 'forked from'
    # Only not 'forked_from' records are kept (with '\N')
    df_chunk = df_chunk[df_chunk['forked_from'].values.astype(str) == '\\N']

    # 2) clean 'deleted'
    df_chunk = df_chunk[df_chunk['deleted'] == 0]

    # 3) clean 'unknown' column
    df_chunk = df_chunk.drop('unknown', 1)

    # 4) re-format the 'url' column to make the links usable
    df_chunk['url'].replace({'api.': ''}, inplace=True, regex=True)
    df_chunk['url'].replace({'repos/': ''}, inplace=True, regex=True)

    # 5) deleted 'forked_from' and 'deleted' columns because they won't be needed for later use
    


    # Output the new CSV file
    df_chunk.to_csv('../dataset/project_test_new.csv', sep=',',mode='a')



# Read the csv file Chunk by Chunk # chunksize =3000

col_names=['id','url','owner_id','name','description','language','created_at','forked_from','deleted','updated_at','unknown']


reader = pd.read_csv('../dataset/projects.csv', skiprows = 1, iterator = True, error_bad_lines = False, names = col_names)

if_loop = True
chunk_size = 1000
number_of_chunk = 0
while if_loop:
    try:
        df_chunk = pd.DataFrame(reader.get_chunk(chunk_size))
        clean(df_chunk)
        number_of_chunk += 1
        if (number_of_chunk == 1 ):
            if_loop = False
            break
    except StopIteration:
        if_loop = False
        print("Iteration is stopped.")

