import duckdb
import polars as pl
from dotenv import load_dotenv
import os
load_dotenv()  


## Only the datasets

dataset_path = r"C:\Users\lucas\Documents\GitHub\Recommind\data\processed"



dataset_ratings = os.path.join(dataset_path, "Books_rating.csv")
dataset_books = os.path.join(dataset_path, "books_data.csv")


books_data = pl.read_csv(dataset_books)
ratings = pl.read_csv(dataset_ratings)

# Register the DataFrames as DuckDB tables
con = duckdb.connect()
con.register('books_data', books_data)
con.register('ratings', ratings)

query = """
SELECT
  books_data.Title,
  books_data.categories,
  books_data.authors
FROM books_data
JOIN ratings
  ON books_data.Title = ratings.Title;
"""

result = con.execute(query).fetchdf()
print(result)


