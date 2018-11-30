# Extract 'description' and 'name' columns
import pandas as pd


col_names=['id','url','owner_id','name','description','language','created_at','updated_at','popularity']
dtypes={'id':str, 'owner_id':str,'popularity':str}
reader = pd.read_csv('../dataset/project_final_v2.csv',encoding='utf-8',names=col_names,dtype=dtypes,skiprows = 1)
reader.drop(columns=['id','url','owner_id','language','created_at','updated_at','popularity'], inplace = True)
reader.to_csv('../dataset/desc.csv',encoding='utf_8_sig')
print("Done")