from recommind_train import Processor, trainer, evaluate_batch_precision_recall
from recommind_model import NCF
from dotenv import load_dotenv
import os
import torch
import joblib
import duckdb
import wandb
import argparse
load_dotenv()  

parser = argparse.ArgumentParser()
parser.add_argument('--NAME', type=str, help='Name of the Experiment')

parser.add_argument('--EXPERIMENT_RUNS', type=int, help='Number of Experiments', default=5)
parser.add_argument('--EPOCHS', type=int, help='Training Epochs')
parser.add_argument('--LEARNING_RATE', type=float, help='Initial Learning Rate', default=0.005)

args = parser.parse_args()

name = args.NAME
lr = args.LEARNING_RATE
epochs = args.EPOCHS
runs = args.EXPERIMENT_RUNS


# Preparing the paths

models_path = os.getenv('models')
wandb_k = os.getenv('wandb_key')

## Only models

encoding_path, ncf_path = os.path.join(models_path, "encoding_models"), os.path.join(models_path, "ncf_model")

def main():

        wandb.login(key=wandb_k)

        # Importing data

        con = duckdb.connect("proto.duckdb")

 
        query = """
        SELECT 
            b.Title, 
            b.authors, 
            b.categories, 
            r.Id, 
            r.User_id, 
            r."review/score",
            b.ratingsCount
        FROM books b
        JOIN ratings r ON b.Title = r.Title
        ORDER BY r.User_id, r.Id, b.categories, b.authors;
        """


        df = con.execute(query).fetchdf()
  
        # Processing 

        proce = Processor(df)

        trainloader, evalloader, n_users, n_items, n_genders, n_authors = proce.run()

        ordinal_encoder = proce.ordinal_encoder

        print(f'Saving the encoders\n')

        joblib.dump(ordinal_encoder, os.path.join(encoding_path, 'ordinal_encoder.joblib'))
        
        artifact = wandb.Artifact(name=f"recommind_{name}", type="encoder")
        artifact.add_file(local_path=os.path.join(encoding_path, 'ordinal_encoder.joblib'), name="encoder_model")
        artifact.save()


        print('Done!\n')

        # Training

        print('The training is starting!\n ')

        model = NCF(n_users, n_items, n_genders, n_authors, n_factors=16)

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
            name_experiment=name,
            n_k = 10,
            total_runs = runs,
            epochs = epochs,
            device='cuda',
            lr=lr,
            weight_decay=1e-5
        )
        

if __name__ == "__main__":
    main()








