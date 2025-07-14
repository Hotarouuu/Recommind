# Data treatment and processing

import polars as pl
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import joblib


class PredictDataset(Dataset):
   
    def __init__(self, dataframe):
        self.X = dataframe.drop('Title', axis=1).to_numpy()
        self.X = torch.tensor(self.X, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

class Encoder:
    def __init__(self, df):
        self.df = df
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.authors_encoder = LabelEncoder()
        self.categories_encoder = LabelEncoder()

    def id_encoder(self):
        encoded = self.ordinal_encoder.fit_transform(self.df.select(['User_id', 'Id']).to_numpy())
        self.df = self.df.with_columns([
            pl.Series('User_id', encoded[:, 0].astype(int)),
            pl.Series('Id', encoded[:, 1].astype(int))
        ])
        self.df = self.df.filter((pl.col('User_id') != -1) & (pl.col('Id') != -1))

    def info_encoder(self):
        authors_encoded = self.authors_encoder.fit_transform(self.df['authors'].to_numpy())
        categories_encoded = self.categories_encoder.fit_transform(self.df['categories'].to_numpy())
        self.df = self.df.with_columns([
            pl.Series('authors', authors_encoded),
            pl.Series('categories', categories_encoded)
        ])

    def transforms(self):
        self.id_encoder()
        self.info_encoder()
        return self.df, self.ordinal_encoder, self.authors_encoder, self.categories_encoder

class Processor:
    def __init__(self, data_path, ratings_path):
        self.data_path = data_path
        self.ratings_path = ratings_path
        self.df_merged = None
        self.n_users = None
        self.n_items = None
        self.n_authors = None
        self.n_categories = None
        self.ordinal_encoder = None
        self.authors_encoder = None
        self.categories_encoder = None

    def data_treatment(self):
        df_data = pl.read_csv(self.data_path)
        df_ratings = pl.read_csv(self.ratings_path)
        df_data = df_data.select(['Title', 'authors', 'categories', 'ratingsCount'])
        df_data = df_data.with_columns(
            pl.col("ratingsCount").fill_null(0),
            pl.col("categories").fill_null("No Category")
        )
        df_ratings = df_ratings.select(['Id', 'Title', 'User_id', 'review/score']).drop_nulls()
        df_merged = df_ratings.join(
            df_data,
            on='Title',
            how='left'
        )
        df_merged = df_merged.with_columns(
            pl.col("categories").str.replace_all('[', '', literal=True).str.replace_all(']', '', literal=True),
            pl.col("authors").str.replace_all('[', '', literal=True).str.replace_all(']', '', literal=True)
        )
        self.df_merged = df_merged

    def encode(self, ordinal_encoder=None, authors_encoder=None, categories_encoder=None):
        if ordinal_encoder is not None and authors_encoder is not None and categories_encoder is not None:
            self.ordinal_encoder = joblib.load(ordinal_encoder)
            self.authors_encoder = joblib.load(authors_encoder)
            self.categories_encoder = joblib.load(categories_encoder)

            encoded = self.ordinal_encoder.transform(self.df_merged.select(['User_id', 'Id']).to_numpy())
            self.df_merged = self.df_merged.with_columns([
                pl.Series('User_id', encoded[:, 0].astype(int)),
                pl.Series('Id', encoded[:, 1].astype(int))
            ])
            self.df_merged = self.df_merged.filter((pl.col('User_id') != -1) & (pl.col('Id') != -1))

            authors_encoded = self.authors_encoder.transform(self.df_merged['authors'].to_numpy())
            categories_encoded = self.categories_encoder.transform(self.df_merged['categories'].to_numpy())
            self.df_merged = self.df_merged.with_columns([
                pl.Series('authors', authors_encoded),
                pl.Series('categories', categories_encoded)
            ])
  

    def reco_user(self, user):
        self.df_merged = self.df_merged.to_pandas()
        user_dataframe = self.df_merged[self.df_merged['User_id'] == user]
        user_items = user_dataframe['Id']
        all_items = self.df_merged['Id'].unique()
        items_to_predict = list(set(all_items) - set(user_items))
        items_to_predict = pd.DataFrame(items_to_predict, columns=['Id'])
        books_data = self.df_merged.drop_duplicates(subset='Id')
        result = pd.merge(books_data, items_to_predict, on='Id', how='inner')
        result = result.drop('review/score', axis=1)
        result['User_id'] = user

        return result, items_to_predict
    
    def pred_user(self, result, top_k : int, book_dict : dict, model, items_to_predict):
        dataset = PredictDataset(result)
        loader = DataLoader(dataset, batch_size=2048, shuffle=True, drop_last=True)

        model.eval()
        scores = []
        with torch.no_grad():
            for X in loader:
                outputs = model(X)
                scores.append(outputs.cpu())
        scores = np.array(scores)
        scores = scores.flatten()
        scores = torch.tensor(scores)

        top_indices = torch.topk(scores, top_k).indices

        top_items = [items_to_predict.iloc[int(i)] for i in top_indices]

        for idx in range(len(top_items)):
            book_id = top_items[idx]['Id']
            book = book_dict[book_id]
            print(f" Recommended Book: {book}")

    def run_pipeline(self, user, ordinal_encoder=None, authors_encoder=None, categories_encoder=None):
        self.data_treatment()
        self.encode(ordinal_encoder, authors_encoder, categories_encoder)
        result, items_to_predict = self.reco_user(user)
        return result, items_to_predict



