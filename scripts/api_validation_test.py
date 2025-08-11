from pydantic import BaseModel, Field, ValidationError
from typing import List
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

query = """SELECT 
    b.Title, 
    b.authors, 
    b.categories, 
    r.Id, 
    r.User_id, 
    r."review/score",
    b.ratingsCount
FROM books b
JOIN ratings r ON b.Title = r.Title;"""

df = con.execute(query).fetchdf()

df = Pipeline.data_treatment(df, enc)

json_data = df.to_dict(orient="records")

payload = {
    'data' : json_data[:10],
    'user' : 212393
}

from typing import Union

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


try:
    obj = WrapperModel.parse_obj(payload)
    print("Validação OK!")
except ValidationError as e:
    print("Erro na validação:")
    print(e.json())
