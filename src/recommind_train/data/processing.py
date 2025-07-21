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

class Processor:
    def __init__(self, database):
        self.df_merged = None
        self.n_users = None
        self.n_items = None
        self.n_authors = None
        self.n_genders = None
        self.ordinal_encoder = None
        self.database = database

    def data_treatment(self):

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

        self.df_merged = self.database.execute(query).fetchdf()
        self.df_merged['categories'] = self.df_merged['categories'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        self.df_merged['authors'] = self.df_merged['authors'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        self.df_merged['categories'] = self.df_merged['categories'].fillna('No Category')
        self.df_merged['ratingsCount'] = self.df_merged['ratingsCount'].fillna(0)
        
    def _encode(self):

        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoded = self.ordinal_encoder.fit_transform(self.df_merged[['User_id', 'Id', 'categories', 'authors']].to_numpy())

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

        # 1. Stratifing
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

        # 2. Train and validation split
        df_train_val = self.df_merged[self.df_merged["User_id"].isin(train_users["User_id"])].copy()

        np.random.seed(42)
        df_train_val['rand'] = df_train_val.groupby('User_id')['User_id'].transform(lambda x: np.random.rand(len(x)))

        df_train = df_train_val[df_train_val['rand'] <= 0.8].drop(columns=['rand'])
        df_val = df_train_val[df_train_val['rand'] > 0.8].drop(columns=['rand'])

        # 3. Cold-start test split
        df_test = self.df_merged[self.df_merged["User_id"].isin(test_users["User_id"])]

        # 4. Transforming the three into dataloaders for Pytorch
        train_dataset = Loader(df_train)
        val_dataset = Loader(df_val)
        test_dataset = Loader(df_test)

        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return trainloader, testloader, valloader

    def run(self):

        self.data_treatment()
        self._encode()
        trainloader, testloader, valloader = self._get_loaders()
        return trainloader, testloader, valloader, self.n_users, self.n_items, self.n_genders, self.n_authors
