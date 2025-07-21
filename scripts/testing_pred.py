# Not useful. It's just for me to test the model

from recommind_pred import Pipeline
from recommind_model import model_config
import torch
import os
from dotenv import load_dotenv
import duckdb
load_dotenv()  

models_path = os.getenv('models')
ncf_path = os.path.join(models_path, "ncf_model")
model = model_config(ncf_path, device='cpu')
encoding_path = os.path.join(models_path, "encoding_models")

# Importing data

con = duckdb.connect("proto.duckdb")

enc = os.path.join(encoding_path, "ordinal_encoder.joblib")

proce = Pipeline(con)

result, items_to_predict = proce.run(212393, enc)


book_df = proce.df_merged[['Id', 'Title']]

book_dict = dict(zip(book_df['Id'], book_df['Title']))

proce.pred_user(result, 10, book_dict, model, items_to_predict)

