# Not useful. It's just for me to test the model

from recommind_pred import Pipeline
from recommind_train import data_treatment
from recommind_model import model_config
import os
from dotenv import load_dotenv
import duckdb
load_dotenv()  

# Preparing the paths

models_path = os.getenv('models')
ncf_path, encoding_path = os.path.join(models_path, "ncf_model"), os.path.join(models_path, "encoding_models")

# Model Config

model = model_config(ncf_path, device='cpu')

# Importing data

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

df = data_treatment(df)

enc = os.path.join(encoding_path, "ordinal_encoder.joblib")

proce = Pipeline(df, model, enc)

proce.run(212393)

result = proce.pred_user(top_k = 10)

print(result)
