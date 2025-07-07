# Data treatment and processing

import polars as pl
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Pytorch

import torch                  
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


def processing(df_merged):

    # 1. Ordinal Encoder
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoded = encoder.fit_transform(df_merged.select(['User_id', 'Id']).to_numpy())

    df_encoded = df_merged.with_columns([
        pl.Series(name='User_id', values=encoded[:, 0].astype(int)),
        pl.Series(name='Id', values=encoded[:, 1].astype(int)),
    ])

    df_encoded = df_encoded.filter(
        (pl.col('User_id') != -1) & (pl.col('Id') != -1)
    )

    n_users = df_encoded.select(pl.col('User_id').max()).item() + 1
    n_items = df_encoded.select(pl.col('Id').max()).item() + 1

    print(f"n_users = {n_users}, n_items = {n_items}")

    # 2. LabelEncoder


    authors = df_encoded["authors"].to_numpy()
    categories = df_encoded["categories"].to_numpy()

    enc_authors = LabelEncoder()
    enc_categories = LabelEncoder()

    authors_encoded = enc_authors.fit_transform(authors)
    categories_encoded = enc_categories.fit_transform(categories)

    # Se quiser colocar de volta no DataFrame
    df_encoded = df_encoded.with_columns([
        pl.Series("authors", authors_encoded),
        pl.Series("categories", categories_encoded)
    ])

    return df_encoded, n_users, n_items

class Loader(Dataset):
    def __init__(self, df):
        super().__init__()
        self.ratings = df

        self.ratings = self.ratings.drop(["Title"], axis=1)


        self.X = self.ratings.drop(['review/score'], axis=1).to_numpy()
        self.y = self.ratings['review/score'].to_numpy()


        self.X, self.y = torch.tensor(self.X, dtype=torch.long), torch.tensor(self.y)
        self.X, self.y = self.X.to(device='cuda'), self.y.to(device='cuda') 

    

    def __getitem__(self, index):
        # Return features and label as tensors
        return self.X[index], self.y[index]

    def __len__(self):
        # Total number of samples
        return len(self.ratings)
    
def data_loader(df):

    df_encoded, n_users, n_items = processing(df)

    n_genders = len(df_encoded['categories'].unique())
    n_authors = len(df_encoded['authors'].unique())

    user_score = df_encoded.groupby("User_id")["review/score"].median().reset_index()

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

    df_train = df_encoded[df_encoded["User_id"].isin(train_users["User_id"])]
    df_val = df_encoded[df_encoded["User_id"].isin(val_users["User_id"])]
    df_test = df_encoded[df_encoded["User_id"].isin(test_users["User_id"])]

    train_dataset = Loader(df_train)
    test_dataset = Loader(df_test)
    eval_dataset = Loader(df_val)

    trainloader = DataLoader(train_dataset, batch_size=2048, shuffle=True, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=2048, shuffle=True, drop_last=True)
    evalloader = DataLoader(eval_dataset, batch_size=2048, shuffle=True, drop_last=True)

    return trainloader, testloader, evalloader, n_users, n_items, n_genders, n_authors
