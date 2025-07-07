from src import data_loader, NeuMF, trainer, evaluate_batch_precision_recall, data_treatment
from dotenv import load_dotenv
import os
import torch

load_dotenv()  

dataset_path = os.getenv("data_dir") # Define variables with .env or add path here

dataset = os.path.join(dataset_path, "processed")
dataset_ratings = os.path.join(dataset, "ratings.csv")
dataset_books = os.path.join(dataset, "books.csv")

df = data_treatment(dataset_books, dataset_ratings)

trainloader, testloader, evalloader, n_users, n_items, n_genders, n_authors = data_loader(df)

# Training






