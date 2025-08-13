# Not useful. It's just for me to test the model

from recommind_pred import Pipeline
import os
from dotenv import load_dotenv
import duckdb
import requests
load_dotenv()  
import numpy as np
import pandas as pd


payload = payload = {"user_id": 'A00117421L76WVWG4UX95'}
# Envia o POST
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload
)

print(f'Status code: {response.status_code}')
print(f'Response: {response.content}')
