from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import duckdb
import pandas as pd
from recommind_pred import Pipeline
from recommind_model import model_config
import os
from dotenv import load_dotenv
load_dotenv()

models_path = os.getenv('models')
ncf_path, encoding_path = os.path.join(models_path, "ncf_model"), os.path.join(models_path, "encoding_models")


print('Loading the database...')
db = duckdb.connect("proto.duckdb")
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
try:
    df = db.execute(query).fetchdf()
except duckdb.Error:
    print('You have to connect to database first. If you already did, please check the .duckdb file.')

print('Done!\n')

print('Loading the model...')
enc = os.path.join(encoding_path, "ordinal_encoder.joblib") # The pipeline class already load the joblib internally
model = model_config(ncf_path, device='cpu')
print('Done!\n')

print(f'Instantiating the pipeline...')
pipe = Pipeline(df, model, enc)
print('Done!\n')

print('Treating the data...')
pipe._data_treatment() 
df = pipe.df_merged
print('Done!\n')

app = FastAPI()

class User(BaseModel):
    id: str

class PredictRequest(BaseModel):
    users: List[User]


@app.get("/")
def read_root():
    return {"Online": "Yes", "model_version": 1}

@app.post("/predict")
def predict(req: PredictRequest):

    if len(req.users) > 1:
        
        result_dict = {}
        
        for user in req.users:
            
            print('User --->', user.id)

            # Encoding the user ID
            user_df = pd.DataFrame([{
                'User_id': user.id.lower(),
                'Id': 'Empty',
                'categories': 'Empty',
                'authors': 'Empty'
            }])

            user_encoded = pipe.ordinal_encoder.transform(user_df[['User_id', 'Id', 'categories', 'authors']])
            result = pipe.run(user_encoded[0][0])

            # Decoding the book IDs
            result_df = pd.DataFrame({
                'User_id': [user_encoded[0][0]] * len(result),
                'Id': result,
                'categories': -1 * len(result),
                'authors': -1 * len(result)
            })

            decoded = pipe.ordinal_encoder.inverse_transform(result_df[['User_id', 'Id', 'categories', 'authors']])
            recommendations = [row[1].upper() for row in decoded]
            result_dict[user.id] = recommendations  

        return {"predictions": result_dict}
    
    else:
        
        # For single user
        user = req.users[0]
        
        # Encoding the user ID
        user_df = pd.DataFrame([{
            'User_id': user.id.lower(),
            'Id': 'Empty',
            'categories': 'Empty',
            'authors': 'Empty'
        }])

        user_encoded = pipe.ordinal_encoder.transform(user_df[['User_id', 'Id', 'categories', 'authors']])
        result = pipe.run(user_encoded[0][0])

        # Decoding the book IDs
        result_df = pd.DataFrame({
            'User_id': [user_encoded[0][0]] * len(result),
            'Id': result,
            'categories': -1 * len(result),
            'authors': -1 * len(result)
        })

        decoded = pipe.ordinal_encoder.inverse_transform(result_df[['User_id', 'Id', 'categories', 'authors']])
        recommendations = [row[1].upper() for row in decoded]

        return {"predictions": recommendations}


