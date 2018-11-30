# The function to extract repositories with more than 100 watchers
import pandas as pd

def merge_extract(df_chunk):

    # Merge the chunk with the corresponding popularity
    df_chunk_new = pd.merge(df_chunk, reader_popularity, left_on='id', right_on='repo_id', how='left')
    df_chunk_new.drop('repo_id', axis=1, inplace=True)

    # Select the repository with more than or equal to 100 watchers
    df_chunk_new = df_chunk_new[ df_chunk_new['popularity'] >= 100 ]

    # Output to CSV
    df_chunk_new.to_csv('../dataset/project_final.csv', sep=',', mode='a', encoding='utf_8_sig', index=False, chunksize=200000)


# Read the csv file Chunk by Chunk

col_names=['id','url','owner_id','name','description','language','created_at','updated_at']
col_names_popularity=['repo_id','popularity']
dtypes ={'id':str, 'owner_id':str}
dtypes_popularity = {'repo_id':str}

reader = pd.read_csv('../dataset/project_clean.csv', skiprows = 1, iterator = True, error_bad_lines = False, names = col_names, encoding='utf-8', dtype = dtypes)
reader_popularity = pd.read_csv('../dataset/popularity_count.csv', skiprows = 1, error_bad_lines = False, names = col_names_popularity, encoding='utf-8', dtype = dtypes_popularity)

if_loop = True
chunk_size = 200000
while if_loop:
    try:
        df_chunk = pd.DataFrame(reader.get_chunk(chunk_size))
        merge_extract(df_chunk)
    except StopIteration:
        if_loop = False
        print("Iteration is stopped.")