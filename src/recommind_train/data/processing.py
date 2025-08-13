# Data treatment and processing

import polars as pl
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
 
def clean_string_columns(df, columns):
    for col in columns:
        df[col] = df[col].astype(str)

        df[col] = (
            df[col]
            .str.replace(r'[\[\]\(\)\'\"]', '', regex=True) 
            .str.replace(r'\s+', ' ', regex=True)           
            .str.strip()                                   
            .str.lower()                                    
        )

    return df

class Loader(Dataset):

    def __init__(self, df):

        super().__init__()
        self.ratings = df
        self.ratings = self.ratings.drop(["Title"], axis=1)
        self.X = self.ratings.drop(['review/score'], axis=1).to_numpy()
        self.y = self.ratings['review/score'].to_numpy()
        self.X, self.y = torch.tensor(self.X, dtype=torch.long), torch.tensor(self.y)

    def __getitem__(self, index):

        return self.X[index], self.y[index]
    
    def __len__(self):

        return len(self.ratings)


class Processor:
    def __init__(self, df_merged):
        self.df_merged = df_merged
        self.n_users = None
        self.n_items = None
        self.n_authors = None
        self.n_genders = None
        self.ordinal_encoder = None

        
    def data_treatment(self, use_encoder=True):

        self.df_merged['categories'] = self.df_merged['categories'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        self.df_merged['authors'] = self.df_merged['authors'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        self.df_merged = clean_string_columns(self.df_merged, ['User_id', 'Id', 'categories', 'authors'])
        self.df_merged['categories'] = self.df_merged['categories'].fillna('No Category')
        self.df_merged['ratingsCount'] = self.df_merged['ratingsCount'].fillna(0)
        self.df_merged['User_id'] = self.df_merged['User_id'].fillna('No User')
        self.df_merged['authors'] = self.df_merged['authors'].fillna('No Authors')

        if use_encoder:
            self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoded = self.ordinal_encoder.fit_transform(self.df_merged[['User_id', 'Id', 'categories', 'authors']])

            self.df_merged['User_id'] = encoded[:, 0].astype(int)
            self.df_merged['Id'] = encoded[:, 1].astype(int)
            self.df_merged['categories'] = encoded[:, 2].astype(int)
            self.df_merged['authors'] = encoded[:, 3].astype(int)
            self.n_users = self.df_merged['User_id'].max() + 1
            self.n_items = self.df_merged['Id'].max() + 1
            self.n_genders = self.df_merged['categories'].max() + 1
            self.n_authors = self.df_merged['authors'].max() + 1

            self.df_merged = self.df_merged[(self.df_merged['User_id'] != -1) & (self.df_merged['Id'] != -1)]

    def _get_loaders(self, batch_size=2048):

        user_counts = self.df_merged['User_id'].value_counts()
        self.df_merged = self.df_merged[self.df_merged['User_id'].isin(user_counts[user_counts >= 2].index)]

        user_score = self.df_merged.groupby("User_id")["review/score"].median().reset_index()

        train_users, val_users = train_test_split(
            user_score,
            test_size=0.2, 
            stratify=user_score["review/score"],
            random_state=42
        )

        df_train_val = self.df_merged[self.df_merged["User_id"].isin(train_users["User_id"])].copy()

        np.random.seed(42)
        df_train_val['rand'] = df_train_val.groupby('User_id')['User_id'].transform(lambda x: np.random.rand(len(x)))

        df_train = df_train_val[df_train_val['rand'] <= 0.8].drop(columns=['rand'])
        df_val = df_train_val[df_train_val['rand'] > 0.8].drop(columns=['rand'])

        # 4. Transforming the three into dataloaders for Pytorch
        train_dataset = Loader(df_train)
        val_dataset = Loader(df_val)

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        return trainloader, valloader

    def run(self):
        self.data_treatment()
        trainloader, valloader = self._get_loaders()
        return trainloader, valloader, self.n_users, self.n_items, self.n_genders, self.n_authors
