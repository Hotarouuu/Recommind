from recommind_packages import Processor, NeuMF, trainer, evaluate_batch_precision_recall, data_treatment
from dotenv import load_dotenv
import os
import torch
import joblib
load_dotenv()  


# Preparing the paths

## Only models
models_path = os.getenv('models')
encoding_path = os.path.join(models_path, "encoding_models")
ncf_path = os.path.join(models_path, "ncf_model")

## Only the datasets

dataset_path = os.getenv("data_dir") 
dataset = os.path.join(dataset_path, "processed")
dataset_ratings = os.path.join(dataset, "Books_rating.csv")
dataset_books = os.path.join(dataset, "books_data.csv")


def main():
  
        # Processing 

        proce = Processor(dataset_books, dataset_ratings)

        trainloader, testloader, evalloader, n_users, n_items, n_genders, n_authors = proce.run()

        ordinal_encoder = proce.ordinal_encoder


        print(f'Saving the encoders')

        joblib.dump(ordinal_encoder, os.path.join(encoding_path, 'ordinal_encoder.joblib'))

        print('Done!\n')

        # Preparing to the training 

        trainloader, testloader, evalloader, n_users, n_items, n_genders, n_authors = proce.run()


        # Training
        print('The training is starting!\n ')

        model = NeuMF(n_users, n_items, n_genders, n_authors, n_factors=16)

        train = trainer(
            model,
            ncf_path,
            trainloader,
            evalloader,
            device='cuda',
            early_stopping=True,
            n_factors=16,
            lr=0.0005,
            weight_decay=1e-5
        )

        config = {
                'n_users': n_users,
                'n_items': n_items,
                'n_genders': n_genders,
                'n_authors': n_authors,
                'n_factors' : 16

        }
        torch.save({
                'model_state_dict': model.state_dict(),
                'config': config
        }, os.path.join(ncf_path, 'recommind_model.pth'))

        
        # Evaluation

        k = 10

        avg_precision, avg_recall, f_score, user_item_scores= evaluate_batch_precision_recall(testloader, model, k=k, device='cuda')

        print(f"Precision@{k}: {avg_precision * 100:.4f}%")
        print(f"Recall@{k}: {avg_recall * 100:.4f}%")
        print(f'F-Score@{k}: {f_score * 100:4f}%')


if __name__ == "__main__":
    main()








