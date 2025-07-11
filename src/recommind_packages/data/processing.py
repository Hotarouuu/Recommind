# Data treatment and processing

import polars as pl
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Pytorch

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
        # Return features and label as tensors
        return self.X[index], self.y[index]

    def __len__(self):
        # Total number of samples
        return len(self.ratings)

class Encoder:
    def __init__(self, df):

        self.df = df
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.authors_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()

    def id_encoder(self):

        encoded = self.ordinal_encoder.fit_transform(self.df.select(['User_id', 'Id']).to_numpy())
        self.df = self.df.with_columns([
            pl.Series('User_id', encoded[:, 0].astype(int)),
            pl.Series('Id', encoded[:, 1].astype(int))
        ])
        self.df = self.df.filter((pl.col('User_id') != -1) & (pl.col('Id') != -1))

    def info_encoder(self):

        authors_encoded = self.authors_encoder.fit_transform(self.df['authors'].to_numpy())
        categories_encoded = self.gender_encoder.fit_transform(self.df['categories'].to_numpy())
        self.df = self.df.with_columns([
            pl.Series('authors', authors_encoded),
            pl.Series('categories', categories_encoded)
        ])

    def transforms(self):

        self.id_encoder()
        self.info_encoder()

        return self.df, self.ordinal_encoder, self.authors_encoder, self.gender_encoder


class Processor():
    def __init__(self, df_merged):
        self.df_merged = df_merged
        self.n_users = 0
        self.n_items = 0
        self.n_authors = 0 
        self.n_genders = 0

    def encoding(self):
    
        encoder = Encoder(self.df_merged)
        self.df_merged, self.ordinal_encoder, self.authors_encoder, self.gender_encoder = encoder.transforms()
        self.n_users = self.df_merged.select(pl.col('User_id').max()).item() + 1
        self.n_items = self.df_merged.select(pl.col('Id').max()).item() + 1
        self.n_genders = len(self.df_merged['categories'].unique())
        self.n_authors = len(self.df_merged['authors'].unique())
        return self.df_merged, self.ordinal_encoder, self.authors_encoder, self.gender_encoder

    def data_loader(self):

        self.df_merged = self.df_merged.to_pandas()

        user_score = self.df_merged.groupby("User_id")["review/score"].median().reset_index()

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

        df_train = self.df_merged[self.df_merged["User_id"].isin(train_users["User_id"])]
        df_val = self.df_merged[self.df_merged["User_id"].isin(val_users["User_id"])]
        df_test = self.df_merged[self.df_merged["User_id"].isin(test_users["User_id"])]

        train_dataset = Loader(df_train)
        test_dataset = Loader(df_test)
        eval_dataset = Loader(df_val)

        trainloader = DataLoader(train_dataset, batch_size=2048, shuffle=True, drop_last=True)
        testloader = DataLoader(test_dataset, batch_size=2048, shuffle=True, drop_last=True)
        evalloader = DataLoader(eval_dataset, batch_size=2048, shuffle=True, drop_last=True)

        return trainloader, testloader, evalloader, self.n_users, self.n_items, self.n_genders, self.n_authors

    def run_pipeline(self):
        self.encoding()
        trainloader, testloader, evalloader, self.n_users, self.n_items, self.n_genders, self.n_authors = self.data_loader()
        return trainloader, testloader, evalloader, self.n_users, self.n_items, self.n_genders, self.n_authors
