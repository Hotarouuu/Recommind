from recommind_packages import Processor, NeuMF, trainer, evaluate_batch_precision_recall, data_treatment
from dotenv import load_dotenv
import os
import torch
import joblib


load_dotenv()  

dataset_path = os.getenv("data_dir") # Define variables with .env or add path here
models_path = os.getenv('models')

dataset = os.path.join(dataset_path, "processed")
dataset_ratings = os.path.join(dataset, "Books_rating.csv")
dataset_books = os.path.join(dataset, "books_data.csv")

encoding_path = os.path.join(models_path, "encoding_models")

print('Initializing training script...\n')

df = data_treatment(dataset_books, dataset_ratings)

proce = Processor(df)

# We need the encoders, because we can't initialize one everytime

_ ,ordinal_encoder, authors_encoder, gender_encoder = proce.encoding()

print(f'Saving the encoders')

joblib.dump(ordinal_encoder, os.path.join(encoding_path, 'ordinal_encoder.joblib'))
joblib.dump(authors_encoder, os.path.join(encoding_path, 'authors_encoder.joblib'))
joblib.dump(gender_encoder, os.path.join(encoding_path, 'gender_encoder.joblib'))

print('Done!\n')

trainloader, testloader, evalloader, n_users, n_items, n_genders, n_authors = proce.run_pipeline()


# Training
print('The training is starting!\n ')

model = NeuMF(n_users, n_items,n_genders, n_authors, device='cuda', n_factors=16)

train = trainer(model, 
        trainloader, 
        evalloader, 
        device = 'cuda', 
        early_stopping = True, 
        n_factors = 16, 
        lr = 0.0005, 
        weight_decay = 1e-5)

# Evaluation

k = 10

avg_precision, avg_recall, f_score, user_item_scores= evaluate_batch_precision_recall(testloader, model, k=k)

print(f"Precision@{k}: {avg_precision * 100:.4f}%")
print(f"Recall@{k}: {avg_recall * 100:.4f}%")
print(f'F-Score@{k}: {f_score * 100:4f}%')





