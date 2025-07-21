from recommind_train import Processor, trainer, evaluate_batch_precision_recall
from recommind_model import NeuMF
from dotenv import load_dotenv
import os
import torch
import joblib
import duckdb
import wandb
load_dotenv()  


# Preparing the paths

## Only models
models_path = os.getenv('models')
encoding_path = os.path.join(models_path, "encoding_models")
ncf_path = os.path.join(models_path, "ncf_model")
wandb_k = os.getenv('wandb_key')

## Only the datasets

dataset_path = os.getenv("data_dir") 
dataset = os.path.join(dataset_path, "processed")
dataset_ratings = os.path.join(dataset, "Books_rating.csv")
dataset_books = os.path.join(dataset, "books_data.csv")


def main():

        wandb.login(key=wandb_k)

        # Importing data

        con = duckdb.connect("scripts/proto.duckdb")
  
        # Processing 

        proce = Processor(con)

        trainloader, testloader, evalloader, n_users, n_items, n_genders, n_authors = proce.run()

        ordinal_encoder = proce.ordinal_encoder

        print(f'Saving the encoders')

        joblib.dump(ordinal_encoder, os.path.join(encoding_path, 'ordinal_encoder.joblib'))

        print('Done!\n')

        # Training

        print('The training is starting!\n ')

        model = NeuMF(n_users, n_items, n_genders, n_authors, n_factors=16)

        config = {
        'n_users': n_users,
        'n_items': n_items,
        'n_genders': n_genders,
        'n_authors': n_authors,
        'n_factors' : 16
        }

        train = trainer(
            model,
            config,
            ncf_path,
            trainloader,
            evalloader,
            testloader,
            n_k = 10,
            total_runs = 5,
            epochs = 1,
            device='cpu',
            early_stopping=True,
            n_factors=16,
            lr=0.0005,
            weight_decay=1e-5
        )
        

if __name__ == "__main__":
    main()








