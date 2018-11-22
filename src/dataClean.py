# Import necessary packages
import pandas as pd


def clean(df_chunk):
    # Read the CSV file for 'Project'
    #ignore the first row because it is a false entry

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


    # Output the new CSV file
    df_chunk.to_csv('../dataset/project_test_1.csv', sep=',',mode='a')


# Read the csv file Chunk by Chunk # chunksize =3000

for chunk in pd.read_csv('../dataset/projects.csv', skiprows=1, names=['id','url','owner_id','name','description','language','created_at','forked_from','deleted','updated_at','unknown'],chunksize=3000):
    clean(chunk)