from fastapi import FastAPI
from pydantic import BaseModel
import duckdb
import pandas as pd
import joblib
from recommind_pred import Pipeline
from recommind_model import model_config
import os
from dotenv import load_dotenv
import numpy as np
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

df = db.execute(query).fetchdf()

print(df['User_id'])

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

class PredictRequest(BaseModel):
    user_id: str  


@app.get("/")
def read_root():
    return {"Online": "Yes", "model_version": 1}

@app.post("/predict")
def predict(req: PredictRequest):
    print(req.user_id)
    user_df = pd.DataFrame([{
        'User_id': req.user_id.lower(),
        'Id': 'Empty',
        'categories': 'Empty',
        'authors': 'Empty'
    }])

    user_encoded = pipe.ordinal_encoder.transform(user_df[['User_id', 'Id', 'categories', 'authors']])
    result = pipe.run(user_encoded[0][0])

    return {"predictions": result.tolist()}

