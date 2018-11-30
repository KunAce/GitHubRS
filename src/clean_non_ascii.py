# read the CSV file and clean entries with non-ascii 'description'

import pandas as pd
import numpy as np

# function to tell if a sentence is english or non-english
def if_english(sentence):
    return all(ord(c) < 128 for c in sentence)


col_names=['id','url','owner_id','name','description','language','created_at','updated_at','popularity']
dtypes={'id':str, 'owner_id':str,'popularity':str}
reader = pd.read_csv('../dataset/project_final_v2.csv',encoding='utf-8',names=col_names,dtype=dtypes,skiprows = 1)


# Loop through each row in the DataFrame
for idx in reader.index:
    if not if_english(reader.get_value(idx,'description')): # get_value deprecated, need modification later
        reader.set_value(idx,'description',np.nan)


reader.dropna(subset=['description'], inplace = True)
reader.to_csv('../dataset/project_final_en.csv',encoding='utf_8_sig')
print("Done")



