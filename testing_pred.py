# Not useful. It's just for me to test the model

from recommind_pred import Processor
from recommind_model import NeuMF
import torch
import os
from dotenv import load_dotenv
load_dotenv()  
models_path = os.getenv('models')

ncf_path = os.path.join(models_path, "ncf_model")



recommind_model = torch.load(os.path.join(ncf_path, 'recommind_model.pth'))
model = NeuMF(**recommind_model['config'])
model.load_state_dict(recommind_model['model_state_dict'])

dataset_path = os.getenv("data_dir") 
dataset = os.path.join(dataset_path, "processed")
dataset_ratings = os.path.join(dataset, "Books_rating.csv")
dataset_books = os.path.join(dataset, "books_data.csv")
encoding_path = os.path.join(models_path, "encoding_models")

enc1, enc2, enc3 = os.path.join(encoding_path, "ordinal_encoder.joblib"), os.path.join(encoding_path, "authors_encoder.joblib"), os.path.join(encoding_path, "gender_encoder.joblib")


proce = Processor(dataset_books, dataset_ratings)

result, items_to_predict = proce.run_pipeline(212393, enc1, enc2, enc3)


book_df = proce.df_merged[['Id', 'Title']]

book_dict = dict(zip(book_df['Id'], book_df['Title']))

proce.pred_user(result, 10, book_dict, model, items_to_predict)

