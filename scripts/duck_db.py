import duckdb
import polars as pl
from dotenv import load_dotenv
import os
load_dotenv()  

path = os.getenv('data_dir') # -> check your .env . If you dont have it, create one with the name ".env"

## Only the datasets

dataset_path = os.path.join(path, 'raw')
dataset_ratings = os.path.join(dataset_path, "Books_rating.csv")
dataset_books = os.path.join(dataset_path, "books_data.csv")
books_data = pl.read_csv(dataset_books)
ratings = pl.read_csv(dataset_ratings)

# Creating database with DUCKDB

con = duckdb.connect("scripts/proto.duckdb")

con.execute(f"""
CREATE OR REPLACE TABLE books AS
SELECT * FROM read_csv_auto('{dataset_books}');
""")

con.execute(f"""
CREATE OR REPLACE TABLE ratings AS
SELECT * FROM read_csv_auto('{dataset_ratings}');
""")


