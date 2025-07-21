# Data treatment and processing

import polars as pl
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import joblib
from recommind_train import Processor

# Loader for PyTorch

class PredictDataset(Dataset):
   
    def __init__(self, dataframe):
        self.X = dataframe.drop('Title', axis=1).to_numpy()
        self.X = torch.tensor(self.X, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

# Pipeline class for recommendation
## This class don't predict running only the pipeline for now. You have to run the "run()" and then you can use "pred_user()"

class Pipeline(Processor):
    def __init__(self, database):
        super().__init__(database)


    def _encode(self, ordinal_encoder=None):

        if ordinal_encoder is not None:
            self.ordinal_encoder = joblib.load(ordinal_encoder)

            categorical_cols = ['User_id', 'Id', 'authors', 'categories']

            encoded = self.ordinal_encoder.fit_transform(self.df_merged[categorical_cols].to_numpy())

            for i, col in enumerate(categorical_cols):
                self.df_merged[col] = encoded[:, i].astype(int)
            self.df_merged = self.df_merged[(self.df_merged['User_id'] != -1) & (self.df_merged['Id'] != -1)]


    def _reco_user(self, user):
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

    def run(self, user, ordinal_encoder):
        self.data_treatment()
        self._encode(ordinal_encoder)
        result, items_to_predict = self._reco_user(user)
        return result, items_to_predict



