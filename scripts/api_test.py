# Not useful. It's just for me to test the model

from recommind_pred import Pipeline
import os
from dotenv import load_dotenv
import duckdb
import requests
load_dotenv()  
import numpy as np
import pandas as pd

models_path = os.getenv('models')

encoding_path = os.path.join(models_path, "encoding_models")

enc = os.path.join(encoding_path, "ordinal_encoder.joblib")

# Conexão e query
con = duckdb.connect("proto.duckdb")

query = """
SELECT 
    b.Title, 
    b.authors, 
    b.categories, 
    r.Id, 
    r.User_id, 
    r."review/score",
    b.ratingsCount
FROM books b
JOIN ratings r ON b.Title = r.Title
ORDER BY r.User_id, r.Id, b.categories, b.authors;
"""

df = con.execute(query).fetchdf()

df['categories'] = df['categories'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
df['authors'] = df['authors'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
df['categories'] = df['categories'].fillna('No Category')
df['ratingsCount'] = df['ratingsCount'].fillna(0)
df['User_id'] = df['User_id'].fillna('No User')
df['authors'] = df['authors'].fillna('No Authors')

def clean_string_columns(df, columns):
    for col in columns:
        df[col] = df[col].astype(str)

        df[col] = (
            df[col]
            .str.replace(r'[\[\]\(\)\'\"]', '', regex=True) 
            .str.replace(r'\s+', ' ', regex=True)           
            .str.strip()                                   
            .str.lower()                                    
        )

    return df
df = clean_string_columns(df, ['User_id', 'Id', 'categories', 'authors'])

df = Pipeline._data_treatment(df, enc)

json_data = df.to_dict(orient="records")

payload = {
    'data' : json_data[:10000],
    'user' : 0
}
# Envia o POST
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload
)

print(f'Status code: {response.status_code}')
print(f'Response: {response.content}')
