# Tiny fix to duplicate rows of column names
import pandas as pd

# Read the csv file

reader = pd.read_csv('../dataset/project_final.csv',encoding='utf-8')
reader.drop_duplicates(keep='first',inplace = True)
reader.to_csv('../dataset/project_final_v2.csv',encoding='utf_8_sig')
print("Done")