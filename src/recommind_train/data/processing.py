# Data treatment and processing

import polars as pl
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

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

class Encoder:
    def __init__(self, df):
        self.df = df
        self.categorical_cols = ['User_id', 'Id', 'authors', 'categories']
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    def fit_transform(self):

        encoded = self.ordinal_encoder.fit_transform(self.df[self.categorical_cols].to_numpy())
        for i, col in enumerate(self.categorical_cols):
            self.df = self.df.with_columns([
                pl.Series(col, encoded[:, i].astype(int))
            ])
        self.df = self.df.filter((pl.col('User_id') != -1) & (pl.col('Id') != -1))

        return self.df, self.ordinal_encoder

    def transform(self, df):

        encoded = self.ordinal_encoder.transform(df[self.categorical_cols].to_numpy())
        for i, col in enumerate(self.categorical_cols):
            df = df.with_columns([
                pl.Series(col, encoded[:, i].astype(int))
            ])
        df = df.filter((pl.col('User_id') != -1) & (pl.col('Id') != -1))

        return df

class Processor:
    def __init__(self, data_path, ratings_path):
        self.data_path = data_path
        self.ratings_path = ratings_path
        self.df_merged = None
        self.n_users = None
        self.n_items = None
        self.n_authors = None
        self.n_genders = None
        self.ordinal_encoder = None
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
            df_data.select(["Title", "authors", "categories", "ratingsCount"]),
            on='Title',
            how='left'
        )
        df_merged = df_merged.with_columns(
            pl.col("categories").str.replace_all('[', '', literal=True).str.replace_all(']', '', literal=True),
            pl.col("authors").str.replace_all('[', '', literal=True).str.replace_all(']', '', literal=True)
        )
        self.df_merged = df_merged
    def encode(self):

        encoder = Encoder(self.df_merged)
        self.df_merged, self.ordinal_encoder = encoder.fit_transform()
        self.n_users = int(self.df_merged.select(pl.col('User_id').max()).item()) + 1
        self.n_items = int(self.df_merged.select(pl.col('Id').max()).item()) + 1
        self.n_genders = len(self.df_merged['categories'].unique())
        self.n_authors = len(self.df_merged['authors'].unique())

        return self.df_merged, self.ordinal_encoder

    def get_loaders(self, batch_size=2048):

        df = self.df_merged.to_pandas()
        user_score = df.groupby("User_id")["review/score"].median().reset_index()
        train_users, temp_users = train_test_split(
            user_score,
            test_size=0.4,
            stratify=user_score["review/score"],
            random_state=42
        )
        val_users, test_users = train_test_split(
            temp_users,
            test_size=0.5,
            stratify=temp_users["review/score"],
            random_state=42
        )
        df_train = df[df["User_id"].isin(train_users["User_id"])]
        df_val = df[df["User_id"].isin(val_users["User_id"])]
        df_test = df[df["User_id"].isin(test_users["User_id"])]

        train_dataset = Loader(df_train)
        val_dataset = Loader(df_val)
        test_dataset = Loader(df_test)

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return trainloader, testloader, valloader

    def run(self):

        self.data_treatment()
        self.df_merged, self.ordinal_encoder = self.encode()
        trainloader, testloader, valloader = self.get_loaders()
        return trainloader, testloader, valloader, self.n_users, self.n_items, self.n_genders, self.n_authors
