from src import Processor, NeuMF, trainer, evaluate_batch_precision_recall, data_treatment
from dotenv import load_dotenv
import os
import torch

load_dotenv()  

dataset_path = os.getenv("data_dir") # Define variables with .env or add path here

dataset = os.path.join(dataset_path, "processed")
dataset_ratings = os.path.join(dataset, "Books_rating.csv")
dataset_books = os.path.join(dataset, "books_data.csv")

df = data_treatment(dataset_books, dataset_ratings)

proce = Processor(df)

# We need the encoders, because we can't initialize one everytime

_ ,ordinal_encoder, authors_encoder, gender_encoder = proce.encoding()

trainloader, testloader, evalloader, n_users, n_items, n_genders, n_authors = proce.run_pipeline()


# Training

model = NeuMF(n_users, n_items,n_genders, n_authors, device, n_factors=8)

train = trainer(model, 
        trainloader, 
        evalloader,
        n_users, 
        n_items, 
        n_genders, 
        n_authors, device = 'cuda', 
        early_stopping = True, 
        n_factors = 16, 
        lr = 0.0005, 
        weight_decay = 1e-5)

# Evaluation


avg_precision, avg_recall, f_score, user_item_scores= evaluate_batch_precision_recall(testloader, model, k=10)

print(f"Precision@{k}: {avg_precision * 100:.4f}%")
print(f"Recall@{k}: {avg_recall * 100:.4f}%")
print(f'F-Score@{k}: {f_score * 100:4f}%')





