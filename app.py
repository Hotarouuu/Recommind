from fastapi import FastAPI
from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from recommind_pred import Pipeline
from recommind_model import model_config
import os
from dotenv import load_dotenv
load_dotenv()

models_path = os.getenv('models')
ncf_path, encoding_path = os.path.join(models_path, "ncf_model"), os.path.join(models_path, "encoding_models")
enc = os.path.join(encoding_path, "ordinal_encoder.joblib") # The pipeline class already load the joblib internally
model = model_config(ncf_path, device='cpu')

app = FastAPI()

class InputData(BaseModel):
    Title: str
    authors: Union[str, int]
    categories: Union[str, int]
    Id: Union[str, int]
    User_id: Union[str, int]
    review_score: float = Field(alias="review/score")
    ratingsCount: float

class WrapperModel(BaseModel):
    data: List[InputData]
    user: int

@app.get("/")
def read_root():
    return {"Online": "Yes", "model_version": 1}

@app.post("/predict")
def predict(payload: WrapperModel):
    

    df = pd.DataFrame([item.dict() for item in payload.data])

    df = df[
        (df['User_id'] != -1) & 
        (df['Id'] != -1) & 
        (df['authors'] != -1) & 
        (df['categories'] != -1)
    ]

    proc = Pipeline(df, model)
    result = proc.run(payload.user)

    result = result.tolist()

    return {"predictions": result}
